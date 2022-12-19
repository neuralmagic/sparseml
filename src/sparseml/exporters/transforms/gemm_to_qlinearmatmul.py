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

import logging

from onnx import ModelProto, helper, numpy_helper

from sparseml.exporters.transforms.onnx_transform import OnnxTransform
from sparseml.exporters.transforms.utils import (
    INITIALIZER_MATCH,
    MatchResult,
    any_of,
    get_quantization_params,
    get_structural_matches,
    optional_node,
    quantize_array,
)
from sparseml.onnx.utils import ONNXGraph, get_node_attributes, get_node_output_nodes


__all__ = ["GemmToQLinearMatMul"]

_LOGGER = logging.getLogger(__name__)


class GemmToQLinearMatMul(OnnxTransform):
    """
    Transforms Gemm nodes to QLinearMatMul.

    NOTE: Does not match if the structure is
    `Gemm -> QuantizeLinear -> DequantizeLinear -> Gemm`

    Transforms
    ```
    |       weight (initializer)
    |         |
    | input   Q
    |   |     |
    | Q/Dq    Dq   optional bias (initializer)
    |     |   |   |
    |        Gemm
    |         |
    |   optional Q/Dq
    ```
    (where `Q` is QuantizeLinear, and `Dq` is DequantizeLinear)

    into

    ```
        input
        |
    QLinearMatMul
        |
       Dq  bias (initializer)
        | |
        Add
    ```
    """

    def transform(self, model: ModelProto) -> ModelProto:
        graph = ONNXGraph(model)
        matches = get_structural_matches(
            graph,
            op_type="Gemm",
            parent_ops=[
                [any_of("QuantizeLinear", "DequantizeLinear")],
                [INITIALIZER_MATCH, "QuantizeLinear", "DequantizeLinear"],
            ],
            children_ops=[
                [
                    any_of("QuantizeLinear", "DequantizeLinear"),
                    optional_node("DequantizeLinear"),
                ]
            ],
        )
        for match in matches:
            gemm_attributes = get_node_attributes(match.node)
            if any(float(attribute) != 1.0 for attribute in gemm_attributes.values()):
                # can only handle Gemm operations without alpha/beta/transB set
                continue

            output_dequant = match.children[0][1]
            if output_dequant is not None:
                output_dequant_child = graph.get_node_single_child(output_dequant)
                if output_dequant_child and output_dequant_child.op_type == "Gemm":
                    # output quant is not a QDQ block for the current Gemm Node but,
                    # the input QDQ block for a new Gemm block this Gemm should be
                    # skipped and processed by _convert_quantizable_gemm_no_activations
                    continue

            self.log_match(match)
            self._transform_match(model, match)

        return model

    def _transform_match(self, model: ModelProto, match: MatchResult):
        gemm_node = match.node
        (input_quant,) = match.parents[0]
        _, weight_quant, weight_dequant = match.parents[1]
        (output_quant, opt_output_dequant) = match.children[0]

        # can fold the input/output quant ops if they are trivial
        fold_input_quant = input_quant.op_type == "DequantizeLinear"
        fold_output_quant = output_quant.op_type == "QuantizeLinear"

        weight_quantize_params = get_quantization_params(
            model, weight_quant, include_target=True
        )
        # sanity check - matching will handle this
        assert weight_quantize_params.target is not None

        # quantize weight
        quantized_weight = quantize_array(
            weight_quantize_params.target,
            weight_quantize_params.scale,
            weight_quantize_params.zero_point,
            weight_quantize_params.zero_point.dtype,
        )
        quantized_weight = quantized_weight.transpose()  # Gemm has implicit transpose
        quantized_weight_name = "{}.weight_quantized".format(gemm_node.name)
        quantized_weight_initializer = numpy_helper.from_array(
            quantized_weight, name=quantized_weight_name
        )
        model.graph.initializer.append(quantized_weight_initializer)

        # get qmatmul inputs and outputs
        qmatmul_input = input_quant.input[0] if fold_input_quant else gemm_node.input[0]
        qmatmul_inputs = [
            qmatmul_input,  # x
            input_quant.input[1],  # x_scale
            input_quant.input[2],  # x_zero_point
            quantized_weight_name,  # w
            weight_quant.input[1],  # w_scale
            weight_quant.input[2],  # w_zero_point
            output_quant.input[1],  # y_scale
            output_quant.input[2],  # y_zero_point
        ]

        # create qmatmul node and add it to graph
        qmatmul_name = f"{gemm_node.name}_quant"
        qmatmul_output = (
            output_quant.output[0] if fold_output_quant else gemm_node.output[0]
        )
        qmatmul_node = helper.make_node(
            "QLinearMatMul",
            qmatmul_inputs,
            [qmatmul_output],
            name=qmatmul_name,
        )
        self.add_node_deferred(qmatmul_node)

        # add bias term following FC in the graph
        if len(gemm_node.input) > 2:
            mm_child = opt_output_dequant if fold_output_quant else output_quant
            qmatmul_output_name = f"{qmatmul_output}_pre_dq"
            dequant_output_name = f"{qmatmul_output}_post_dq"
            if mm_child is not None and mm_child.op_type == "DequantizeLinear":
                # create hidden output layer for bias add
                add_output_name = mm_child.output[0]
                mm_child.output[0] = dequant_output_name
            else:
                # inject dequantize op for matmul
                qmatmul_node.output[0] = qmatmul_output_name
                mm_child = helper.make_node(
                    "DequantizeLinear",
                    [
                        qmatmul_output_name,  # input
                        output_quant.input[1],  # scale
                        output_quant.input[2],  # zero point
                    ],
                    [dequant_output_name],
                    name=f"{qmatmul_name}_injected_dq",
                )
                self.add_node_deferred(mm_child)
                add_output_name = qmatmul_output  # original qmatmul output name

            # inject bias op for dequantized matmul output
            self.add_node_deferred(
                helper.make_node(
                    "Add",
                    # [add_input, gemm bias]
                    [dequant_output_name, gemm_node.input[2]],
                    [add_output_name],
                    f"{gemm_node.name}_injected_bias_add",
                )
            )

        # Clean up
        self.delete_node_deferred(weight_dequant)
        self.delete_node_deferred(weight_quant)
        if fold_input_quant and len(get_node_output_nodes(model, input_quant)) <= 1:
            self.delete_node_deferred(input_quant)
        if fold_output_quant:
            self.delete_node_deferred(output_quant)
        self.delete_node_deferred(gemm_node)
