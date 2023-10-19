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
from onnx import ModelProto, numpy_helper
import numpy

from sparseml.exporters.transforms import OnnxTransform
from sparseml.exporters.transforms.utils import (
    INITIALIZER_MATCH,
    MatchResult,
    get_quantization_params,
    get_structural_matches,
    optional_node,
    quantize_array,
)
from sparseml.onnx.utils import ONNXGraph


__all__ = ["MatMulToDynamicMatMulIntegerAddCastMul"]


class MatMulToDynamicMatMulIntegerAddCastMul(OnnxTransform):
    """
    A transform that attempts, if possible, to convert MatMul nodes into
    their quantized representation.
    This MatMul is the result of quantizing native torch.matmul using QATMatMul

    NOTE: we expect that INPUT_0 and INPUT_1 are not initializers (in case where
    both inputs are activations)


    Transforms
    ```
    |   scale   input
    |       |    |
    |       |-- Div    zero_point
    |       |    |        |
    |       |   Add ------|
    |       |    |
    |       |  Round    weight (initializer)
    |       |    |        |
    |       |   Clip      Q
    |       |    |        |
    |       |   Sub      DQ
    |       |    |        |
    |       |-- Mul   Transpose (optional)
    |            |        |
    |          MatMul ----|
    ```

    into

    ```
    |   scale   input
    |     |      |
    |     |---- Div            zero_point
    |     |      |                 |
    |     |     Add ---------------|
    |     |      |                 |
    |     |    Round               |
    |     |      |                 |
    |     |     Clip               |
    |     |      |                 |
    |     |     Add (UINT8 shift) Add(UINT8 shift)
    |     |      |                 |
    |     |    Cast (to UINT8)   Cast (to UINT8)
    |     |      |                 |
    |     |  MatMulInteger        Mul
    |     |      |                 |
    |     |     Sub ---------------|
    |     |      |
    |     |    Cast (to FP32)
    |     |      |
    |     |     Mul (weight scale)
    |     |      |
    |     |---- Mul (activation scale)
    ```
    """

    def transform(self, model: ModelProto) -> ModelProto:
        graph = ONNXGraph(model)

        parent_ops = [
            [
                "Div",
                "Add",
                "Round",
                "Clip",
                "Sub",
                "Mul",
            ],
            [
                # weight should be initializer
                INITIALIZER_MATCH,
                "QuantizeLinear",
                "DequantizeLinear",
                optional_node("Transpose"),
            ],
        ]

        matches = []
        for order in range(2):
            matches_ = get_structural_matches(
                graph,
                op_type="MatMul",
                parent_ops=parent_ops,
                children_ops=[[optional_node("Add")]],
            )
            if order == 0:
                # reverse the parent order (weights on first input)
                # before second pass
                parent_ops.reverse()
            else:
                # if matched using reverse order, need to
                # revert parents order to match transformation logic
                for m in matches_:
                    m.parents.reverse()
            matches.extend(matches_)

        processed_div_nodes = []
        for match in matches:
            scale_div_node = match.parents[0][0]
            if scale_div_node in processed_div_nodes:
                new_quantization = False
            else:
                new_quantization = True
                processed_div_nodes.append(scale_div_node)
            self.log_match(match)
            self._transform_match(model, match, new_quantization)
        return model

    def _transform_match(
        self,
        model: ModelProto,
        match: MatchResult,
        new_quantization: bool,
    ):
        (
            scale_div,
            zero_point_add,
            round_node,
            clip,
            sub_node,
            scale_mul,
        ) = match.parents[0]
        weight_init, weight_quant, weight_dequant, opt_transpose = match.parents[1]

        # Find scale and input tensor
        # Scale is the only input in common between the div and the mul nodes
        for inp in scale_div.input:
            if inp in scale_mul.input:
                scale = inp
                break

        # Find zero point
        for inp in zero_point_add.input:
            if inp not in scale_div.output:
                zero_point = inp
                break

        # Quantize weights
        (
            quantized_weight,
            quantized_weight_name,
            weight_scale_name,
            weight_zero_point_name,
            zero_point_shift_name,
        ) = _quantize_weights(model, match.node.name, weight_quant, opt_transpose)

        # If first time quantizing a specific activation tensor.
        # create nodes needed to cast zero point and activation to
        # adequate data types
        if new_quantization:
            _, _, zero_point_cast_node = _cast(zero_point, zero_point, numpy.float32, "INT32")
            self.add_node_deferred(zero_point_cast_node)

            (
                input_add_initializer,
                input_add_cast_node,
                input_cast_node,
            ) = _cast(clip.name, clip.output[0], numpy.float32, "UINT8")
            model.graph.initializer.append(input_add_initializer)
            self.add_node_deferred(input_add_cast_node)
            self.add_node_deferred(input_cast_node)

        # Create MatMulInteger node
        zero_initializer = numpy_helper.from_array(
            numpy.array(128, dtype=numpy.uint8),
            name=f"{match.node.name}_zero",
        )
        model.graph.initializer.append(zero_initializer)
        matmul_node = onnx.helper.make_node(
            "MatMulInteger",
            [
                f"{clip.name}_cast_output",
                quantized_weight_name,
                f"{match.node.name}_zero",
                weight_zero_point_name,
            ],
            [f"{match.node.name}_quant_output"],
            f"{match.node.name}_quant",
        )
        self.add_node_deferred(matmul_node)

        # Create nodes needed to account for the zero point shift
        # First multiply zero point by weights (summed over inner axis)
        zero_point_shift_mul_node = onnx.helper.make_node(
            "Mul",
            [
                f"{zero_point}_cast_output",
                zero_point_shift_name,
            ],
            [f"{match.node.name}_zero_point_shift_mul_output"],
            f"{match.node.name}_zero_point_shift_mul",
        )
        self.add_node_deferred(zero_point_shift_mul_node)

        # Second subtract from the matmul outputs
        zero_point_shift_sub_node = onnx.helper.make_node(
            "Sub",
            [
                matmul_node.output[0],
                f"{match.node.name}_zero_point_shift_mul_output",
            ],
            [f"{match.node.name}_zero_point_shift_sub_output"],
            f"{match.node.name}_zero_point_shift_sub",
        )
        self.add_node_deferred(zero_point_shift_sub_node)

        # Cast output to FP32
        cast_node = onnx.helper.make_node(
            "Cast",
            zero_point_shift_sub_node.output,
            [f"{match.node.name}_output_cast_output"],
            f"{match.node.name}_output_cast",
            to=getattr(onnx.TensorProto, "FLOAT"),  # get Float32 enum id
        )
        self.add_node_deferred(cast_node)

        # Rescale to FP32 range
        # Multiply by weight scale
        weight_scale_node = onnx.helper.make_node(
            "Mul",
            [f"{match.node.name}_output_cast_output", weight_scale_name],
            [f"{match.node.name}_weight_scale_output"],
            f"{match.node.name}_weight_scale",
        )
        self.add_node_deferred(weight_scale_node)

        # Multiply by activation scale
        activation_scale_node = onnx.helper.make_node(
            "Mul",
            [f"{match.node.name}_weight_scale_output", scale],
            match.node.output,
            f"{match.node.name}_activation_scale",
        )
        self.add_node_deferred(activation_scale_node)

        # Clean up
        if new_quantization:
            self.delete_node_deferred(sub_node)
            self.delete_node_deferred(scale_mul)

        self.delete_node_deferred(weight_quant)
        self.delete_node_deferred(weight_dequant)
        if opt_transpose is not None:
            self.delete_node_deferred(opt_transpose)
        self.delete_node_deferred(match.node)


def _cast(prefix, tensor, input_dtype, output_dtype):
    if output_dtype == "UINT8":
        # Create add node to shift values by 128
        add_initializer = numpy_helper.from_array(
            numpy.array(128, dtype=input_dtype),
            name=f"{prefix}_cast_add_initializer",
        )

        add_node = onnx.helper.make_node(
            "Add",
            [tensor, f"{prefix}_cast_add_initializer"],
            [f"{prefix}_add_cast_output"],
            f"{prefix}_add_cast",
        )

        current_tensor = f"{prefix}_add_cast_output"
    else:
        add_initializer = None
        add_node = None
        current_tensor = tensor

    # Create node to cast
    cast_node = onnx.helper.make_node(
        "Cast",
        [current_tensor],
        [f"{prefix}_cast_output"],
        f"{prefix}_cast",
        to=getattr(onnx.TensorProto, output_dtype),
    )

    return add_initializer, add_node, cast_node


def _quantize_weights(model, prefix, weight_quant, opt_transpose):
    # Find weight quantization parameters
    weight_quantize_params = get_quantization_params(
        model, weight_quant, include_target=True
    )
    # sanity check - matching handles this
    assert weight_quantize_params.target is not None

    # Quantize weights and add to graph
    quantized_weight = quantize_array(
        weight_quantize_params.target,
        weight_quantize_params.scale,
        weight_quantize_params.zero_point,
        weight_quantize_params.zero_point.dtype,
    )
    if opt_transpose is not None:
        quantized_weight = quantized_weight.transpose()

    quantized_weight_name = f"{prefix}.weight_quantized"
    quantized_weight_initializer = numpy_helper.from_array(
        quantized_weight, name=quantized_weight_name
    )
    model.graph.initializer.append(quantized_weight_initializer)

    # Add initializers for weight scale and zero_point
    weight_scale_name = f"{prefix}.weight_scale"
    weight_scale_initializer = numpy_helper.from_array(
        weight_quantize_params.scale,
        name=weight_scale_name,
    )
    model.graph.initializer.append(weight_scale_initializer)

    weight_zero_point_name = f"{prefix}.weight_zero_point"
    weight_zero_point_initializer = numpy_helper.from_array(
        weight_quantize_params.zero_point,
        name=weight_zero_point_name,
    )
    model.graph.initializer.append(weight_zero_point_initializer)

    # Add initializer for weight contribution to zero point offset in multiplication
    zero_point_shift = numpy.sum(quantized_weight, axis=0, keepdims=True).astype(numpy.int32)
    zero_point_shift_name = f"{prefix}.zero_point_shift"
    zero_point_shift_initializer = numpy_helper.from_array(
        zero_point_shift,
        name=zero_point_shift_name,
    )
    model.graph.initializer.append(zero_point_shift_initializer)

    return quantized_weight, quantized_weight_name, weight_scale_name, weight_zero_point_name, zero_point_shift_name