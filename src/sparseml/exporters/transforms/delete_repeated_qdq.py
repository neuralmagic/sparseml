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

from onnx import ModelProto

from sparseml.exporters.transforms.onnx_transform import OnnxTransform
from sparseml.exporters.transforms.utils import assert_node_type
from sparseml.onnx.utils import ONNXGraph


__all__ = ["DeleteRepeatedQdq"]


class DeleteRepeatedQdq(OnnxTransform):
    """
    Transforms
    ```
    QuantizeLinear
        |
    DequantizeLinear
        |
    QuantizeLinear
        |
    DequantizeLinear
    ```
    Into
    ```
    QuantizeLinear
        |
    DequantizeLinear
    ```
    """

    def transform(self, model: ModelProto) -> ModelProto:
        graph = ONNXGraph(model)
        nodes_to_delete = []
        quant_nodes = [n for n in model.graph.node if n.op_type == "QuantizeLinear"]
        for quant_node_1 in quant_nodes:
            dequant_node_1 = graph.get_node_single_child(quant_node_1)
            if not assert_node_type(dequant_node_1, "DequantizeLinear"):
                continue
            quant_node_2 = graph.get_node_single_child(dequant_node_1)
            if not assert_node_type(quant_node_2, "QuantizeLinear"):
                continue
            dequant_node_2 = graph.get_node_single_child(quant_node_2)
            if not assert_node_type(dequant_node_2, "DequantizeLinear"):
                continue

            # forward first qat block input to that of the second
            quant_node_2.input[0] = quant_node_1.input[0]

            # remove repeated quant/dequant block
            nodes_to_delete.append(quant_node_1)
            nodes_to_delete.append(dequant_node_1)

        if len(nodes_to_delete) > 0:
            graph.delete_nodes(nodes_to_delete)
        graph.update()
        graph.delete_unused_initializers()
        return model
