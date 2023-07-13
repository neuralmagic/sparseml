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

import numpy
import onnx
from onnx import ModelProto, numpy_helper

from sparseml.exporters.transforms import OnnxTransform
from sparseml.exporters.transforms.utils import (
    MatchResult,
    get_quantization_params,
    get_structural_matches,
)
from sparseml.onnx.utils import ONNXGraph


__all__ = ["MatMulToMatMulIntegerCastMul"]


class MatMulToMatMulIntegerCastMul(OnnxTransform):
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
    ```
    (where `Q` is QuantizeLinear, and `Dq` is DequantizeLinear)

    into

    ```
    | input_0   input_1
    |     |     |
    |     Q     Q
    |     |     |
    |   MatMulInteger
    |       |
    |     Cast (int32 -> float32)
    |       |
    |      Mul (rescale from deleted dequantize)
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
        )
        for match in matches:
            is_parameterized = False
            for quantize_linear_parent in [match.parents[0][0], match.parents[1][0]]:
                if graph.get_init_by_name(quantize_linear_parent.input[0]):
                    is_parameterized = True
            if is_parameterized:
                continue
            self.log_match(match)
            self._do_transform(model, match)
        return model

    def _do_transform(self, model: ModelProto, match: MatchResult):
        a_quant, a_dequant = match.parents[0]
        b_quant, b_dequant = match.parents[1]

        input_quantize_params = get_quantization_params(model, a_dequant)
        weight_quantize_params = get_quantization_params(model, b_dequant)

        # construct inputs for the new `MatMulInteger` node
        qmatmul_inputs = [
            a_dequant.input[0],  # a
            b_dequant.input[0],  # b
            a_dequant.input[2],  # a_zero_point
            b_dequant.input[2],  # b_zero_point
        ]

        # create qmatmul node and add it to graph
        integer_node = onnx.helper.make_node(
            "MatMulInteger",
            qmatmul_inputs,
            [f"{match.node.name}_quant"],
            f"{match.node.name}_quant",
        )
        self.add_node_deferred(integer_node)

        # create Cast node and add it to graph
        cast_node = onnx.helper.make_node(
            "Cast",
            [integer_node.output[0]],
            [f"{integer_node.output[0]}_cast"],
            f"{integer_node.output[0]}_cast",
            to=getattr(onnx.TensorProto, "FLOAT"),  # get Float32 enum id
        )
        self.add_node_deferred(cast_node)

        # create output scale initializer for rescale mul op
        output_scale = input_quantize_params.scale * weight_quantize_params.scale
        quantized_output_scale_name = f"{match.node.name}.quant.output.scale"
        output_scale_initializer = numpy_helper.from_array(
            numpy.asarray(output_scale), name=quantized_output_scale_name
        )
        model.graph.initializer.append(output_scale_initializer)

        # create Mul node for rescale
        mul_node = onnx.helper.make_node(
            "Mul",
            [cast_node.output[0], output_scale_initializer.name],
            [match.node.output[0]],
            f"{cast_node.output[0]}_rescale_mul",
        )
        self.add_node_deferred(mul_node)

        self.delete_node_deferred(a_dequant)
        self.delete_node_deferred(b_dequant)
        self.delete_node_deferred(match.node)
