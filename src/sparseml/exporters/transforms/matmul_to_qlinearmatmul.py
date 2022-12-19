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
from onnx import ModelProto

from sparseml.exporters.transforms import OnnxTransform
from sparseml.exporters.transforms.utils import (
    MatchResult,
    get_structural_matches,
    optional_node,
)
from sparseml.onnx.utils import ONNXGraph


__all__ = ["MatMulToQLinearMatMul"]


class MatMulToQLinearMatMul(OnnxTransform):
    """
    A transform that attempts, if possible, to convert MatMul nodes into
    their quantized representation.
    This MatMul is the result of quantizing native torch.matmul using QATMatMul

    NOTE: we expect that INPUT_0 and INPUT_1 are not initializers (in case where
    both inputs are activations)


    Transforms
    ```
    | input_0   input_1
    |     |     |
    |     Q     Q
    |     |     |
    |     Dq    Dq
    |     |     |
    |     MatMul
    |       |
    | optional Transpose
    |       |
    | optional Reshape
    |       |
    |       Q
    |       |
    |       Dq
    ```
    (where `Q` is QuantizeLinear, and `Dq` is DequantizeLinear)

    into

    ```
    | input_0   input_1
    |     |     |
    |     Q     Q
    |     |     |
    |  QLinearMatMul
    |       |
    | optional Transpose
    |       |
    | optional Reshape
    |       |
    |       Dq
    ```
    """

    def transform(self, model: ModelProto) -> ModelProto:
        graph = ONNXGraph(model)
        matches = get_structural_matches(
            graph,
            parent_ops=[
                ["QuantizeLinear", "DequantizeLinear"],
                ["QuantizeLinear", "DequantizeLinear"],
            ],
            op_type="MatMul",
            children_ops=[
                [
                    optional_node("Transpose"),
                    optional_node("Reshape"),
                    "QuantizeLinear",
                    "DequantizeLinear",
                ]
            ],
        )
        for match in matches:
            for quantize_linear_parent in [match.parents[0][0], match.parents[1][0]]:
                if graph.get_init_by_name(quantize_linear_parent.input[0]):
                    continue
            self.log_match(match)
            self._do_transform(model, match)
        return model

    def _do_transform(self, model: ModelProto, match: MatchResult):
        a_quant, a_dequant = match.parents[0]
        b_quant, b_dequant = match.parents[1]
        opt_transpose, opt_reshape, output_quant, output_dequant = match.children[0]

        # construct inputs for the new `QLinearMatmul` node
        qmatmul_inputs = [
            a_dequant.input[0],  # a
            a_dequant.input[1],  # a_scale
            a_dequant.input[2],  # a_zero_point
            b_dequant.input[0],  # b
            b_dequant.input[1],  # b_scale
            b_dequant.input[2],  # b_zero_point
            output_quant.input[1],  # y_scale
            output_quant.input[2],  # y_zero_point
        ]

        # set dequantize's input to quant's input
        # NOTE: this handles the presence of the optional transpose/reshape nodes
        output_dequant.input[0] = output_quant.input[0]

        # create qmatmul node and add it to graph
        self.add_node_deferred(
            onnx.helper.make_node(
                "QLinearMatMul",
                qmatmul_inputs,
                [match.node.output[0]],
                "{}_quant".format(match.node.name),
            )
        )

        self.delete_node_deferred(a_dequant)
        self.delete_node_deferred(b_dequant)
        self.delete_node_deferred(output_quant)
        self.delete_node_deferred(match.node)
