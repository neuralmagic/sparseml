# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import onnx
from onnx import ModelProto, NodeProto

from sparseml.exporters.transforms.onnx_transform import OnnxTransform
from sparseml.onnx.utils import ONNXGraph, get_quantize_parent_for_dequantize_node


__all__ = ["QuantizeResiduals"]


class QuantizeResiduals(OnnxTransform):
    """
    To avoid storing the identity value of a ResNet block in fp32, this optimization
    will pass the identity value through the same quantize operation as the ResNet
    block and add a de-quantize operation for the identity before the add.
    Function will match to any add operation whose inputs are the output of a relu
    or add op and a quantize -> de-quantize block that takes the same relu as input.
    Performs this optimization in place.

    Transforms
    ```
    | Relu or Add
    | |       |
    | Q       |
    | |       |
    | (any)   |
    | |       |
    | Dq      |
    | |       |
    |   Add
    ```
    (where `Q` is QuantizeLinear, `Dq` is DequantizeLinear, `(any)` means any sub graph)

    into
    ```
    | Relu or Add
    |     |
    |     Q
    | |       |
    | (any)   |
    | |       |
    | Dq      Dq
    | |       |
    |   Add
    ```
    """

    def transform(self, model: ModelProto) -> ModelProto:
        graph = ONNXGraph(model)
        add_nodes = [node for node in model.graph.node if node.op_type == "Add"]
        for add_node in add_nodes:
            add_inputs = [
                i for i in graph.get_node_parents(add_node) if isinstance(i, NodeProto)
            ]
            if len(add_inputs) != 2:
                continue
            # extract dequantize input and relu/add input
            dequantize_node = [i for i in add_inputs if i.op_type == "DequantizeLinear"]
            other_input_node = [i for i in add_inputs if i.op_type in ["Add", "Relu"]]
            if not dequantize_node or not other_input_node:  # pattern not matched
                continue
            dequantize_node = dequantize_node[0]  # unwrap
            other_input_node = other_input_node[0]  # unwrap

            quantize_node = get_quantize_parent_for_dequantize_node(
                model, dequantize_node
            )

            # check that the quantize block takes input from the same relu
            if (
                quantize_node is None
                or quantize_node.input[0] != other_input_node.output[0]
            ):
                continue

            self.log_match(add_node)

            # create de-quantize node for identity
            dequant_output = f"{other_input_node.output[0]}_identity_dequantized"
            identity_dequantize_node = onnx.helper.make_node(
                "DequantizeLinear",
                [quantize_node.output[0]] + quantize_node.input[1:],
                [dequant_output],
                f"{other_input_node.output[0]}_identity_dequantized",
            )
            self.add_node_deferred(identity_dequantize_node)

            # swap the relu input for the de-quantized identity in the add
            relu_input_idx = [
                i
                for i, inp in enumerate(add_node.input)
                if inp == other_input_node.output[0]
            ][0]
            add_node.input[relu_input_idx] = dequant_output
        return model
