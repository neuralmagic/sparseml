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

from onnx import ModelProto, helper, numpy_helper

from sparseml.exporters.transforms.onnx_transform import OnnxTransform
from sparseml.exporters.transforms.utils.helpers import (
    QUANTIZE_OP_NAMES,
    delete_quant_node,
    get_quantization_params,
    quantize_array,
)
from sparseml.exporters.transforms.utils.matching import (
    INITIALIZER_MATCH,
    MatchResult,
    get_structural_matches,
    optional_node,
)
from sparseml.onnx.utils.graph_editor import (
    ONNXGraph,
    remove_node_and_params_from_graph,
)
from sparseml.onnx.utils.helpers import get_node_attributes, get_node_output_nodes


__all__ = ["GemmToQLinearMatMul"]


class GemmToQLinearMatMul(OnnxTransform):
    """
    # Variant 1

    ```
                        weight (initializer)
                        |
    input               QuantizeLinear
    |                   |
    DequantizeLinear    DequantizeLinear
    |                   |
            Gemm ( with no attributes)
            |
            QuantizeLinear
            |
            NOT (DequantizeLinear -> Gemm)
    ```

    into

    `input -> QLinearMatmul`

    # Variant 2 (with bias)

    ```
                        weight (initializer)
                        |
    input               QuantizeLinear
    |                   |
    DequantizeLinear    DequantizeLinear    bias
    |                   |                   |
            Gemm ( with no attributes)
            |
            QuantizeLinear
            |
            NOT (DequantizeLinear -> Gemm)
    ```

    into

    ```
    input
    |
    QLinearMatMul
    |
    DequantizeLinear    bias
    |                   |
                Add
    ```

    # Variant 3 (with bias & DequantizeLinear)

    ```
                        weight (initializer)
                        |
    input               QuantizeLinear
    |                   |
    DequantizeLinear    DequantizeLinear    bias
    |                   |                   |
            Gemm ( with no attributes)
            |
            QuantizeLinear
            |
            DequantizeLinear -> (NOT -> Gemm)
    ```

    into

    ```
    input
    |
    QLinearMatMul
    |
    DequantizeLinear    bias
    |                   |
                Add
    ```
    """

    def transform(self, model: ModelProto) -> ModelProto:
        graph = ONNXGraph(model)
        matches = get_structural_matches(
            graph,
            op_type="Gemm",
            parent_ops=[
                [],
                [INITIALIZER_MATCH, "QuantizeLinear", "DequantizeLinear"],
            ],
            children_ops=[[]],
        )
        for match in matches:
            gemm_attributes = get_node_attributes(match.node)
            if any(float(attribute) != 1.0 for attribute in gemm_attributes.values()):
                # can only handle Gemm operations without alpha/beta/transB set
                continue

            input_quant = graph.get_node_single_parent(match.node, 0)
            if not input_quant or input_quant.op_type not in QUANTIZE_OP_NAMES:
                continue
            match.parents[0].append(input_quant)

            output_quant = graph.get_node_single_child(match.node)
            if not output_quant or output_quant.op_type not in QUANTIZE_OP_NAMES:
                continue
            match.children[0].append(output_quant)

            output_dequant = graph.get_node_single_child(output_quant)
            if output_dequant and output_dequant.op_type in QUANTIZE_OP_NAMES:
                match.children[0].append(output_dequant)
                output_dequant_child = graph.get_node_single_child(output_dequant)
                if output_dequant_child and output_dequant_child.op_type == "Gemm":
                    # output quant is not a QDQ block for the current Gemm Node but,
                    # the input QDQ block for a new Gemm block this Gemm should be
                    # skipped and processed by _convert_quantizable_gemm_no_activations
                    continue
            else:
                match.children[0].append(None)

            self._do_transform(model, match)

        graph = ONNXGraph(model)
        graph.sort_nodes_topologically()
        graph.delete_unused_initializers()
        return model

    def _do_transform(self, model: ModelProto, match: MatchResult):
        gemm_node = match.node
        (input_quant,) = match.parents[0]
        _, weight_quant, weight_dequant = match.parents[1]
        (output_quant, opt_output_dequant) = match.children[0]

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
        qmatmul_inputs = [
            input_quant.input[0],  # x
            input_quant.input[1],  # x_scale
            input_quant.input[2],  # x_zero_point
            quantized_weight_name,  # w
            weight_quant.input[1],  # w_scale
            weight_quant.input[2],  # w_zero_point
            output_quant.input[1],  # y_scale
            output_quant.input[2],  # y_zero_point
        ]

        qmatmul_name = f"{gemm_node.name}_quant"
        qmatmul_output = output_quant.output[0]

        # create qmatmul node and add it to graph
        qmatmul_node = helper.make_node(
            "QLinearMatMul",
            qmatmul_inputs,
            [qmatmul_output],
            name=qmatmul_name,
        )
        model.graph.node.append(qmatmul_node)

        # add bias term following FC in the graph
        if len(gemm_node.input) > 2:
            qmatmul_output_name = f"{qmatmul_output}_pre_dq"
            dequant_output_name = f"{qmatmul_output}_post_dq"
            if opt_output_dequant is not None:
                # sanity check
                assert opt_output_dequant.op_type == "DequantizeLinear"

                # create hidden output layer for bias add
                add_output_name = opt_output_dequant.output[0]
                opt_output_dequant.output[0] = dequant_output_name
            else:
                # inject dequantize op for matmul
                model.graph.node[-1].output[0] = qmatmul_output_name
                opt_output_dequant = helper.make_node(
                    "DequantizeLinear",
                    [
                        qmatmul_output_name,  # input
                        output_quant.input[1],  # scale
                        output_quant.input[2],  # zero point
                    ],
                    [dequant_output_name],
                    name=f"{qmatmul_name}_injected_dq",
                )
                model.graph.node.append(opt_output_dequant)
                add_output_name = qmatmul_output  # original qmatmul output name

            # inject bias op for dequantized matmul output
            qmatmul_bias_add_node = helper.make_node(
                "Add",
                [
                    dequant_output_name,  # add input
                    gemm_node.input[2],  # Gemm bias
                ],
                [add_output_name],
                f"{gemm_node.name}_injected_bias_add",
            )
            model.graph.node.append(qmatmul_bias_add_node)

        # delete folded quantization ops
        delete_quant_node(model, weight_dequant)
        delete_quant_node(model, weight_quant)
        if len(get_node_output_nodes(model, input_quant)) <= 1:
            # fold if this gemm is the only node that reads from this quant op
            delete_quant_node(model, input_quant)
        delete_quant_node(model, output_quant)

        # delete original Gemm node
        remove_node_and_params_from_graph(model, gemm_node)
