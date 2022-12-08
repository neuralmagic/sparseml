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

import onnx
from onnx import ModelProto

from sparseml.exporters.transforms import OnnxTransform
from sparseml.exporters.transforms.utils import (
    delete_quant_node,
    get_structural_matches,
    optional_node,
)
from sparseml.onnx.utils import ONNXGraph, remove_node_and_params_from_graph


__all__ = ["convert_matmul_to_quantized"]

_LOGGER = logging.getLogger(__name__)


def convert_matmul_to_quantized(
    match: "MatchResult", model: ModelProto  # noqa F821
) -> ModelProto:
    """
    Converts a matmul node to a quantized matmul node
    and updates the onnx model accordingly

    :param match: Match node (matmul) to be quantized
    :param model: ONNX model to be transformed
    :return: ONNX model with the quantized matmul node
    """
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

    # remove the `DequantizeLinear` nodes that precede the `MatMul` node,
    # as well as the `QuantizeLinear` node that follows the `MatMul` node
    # (or follows the last optional node if optional nodes present)
    for node_to_delete in [a_dequant, b_dequant, output_quant]:
        delete_quant_node(model, node_to_delete)

    # set dequantize's input to quant's input
    # NOTE: this handles the presence of the optional transpose/reshape nodes
    output_dequant.input[0] = output_quant.input[0]

    # create qmatmul node and add it to graph
    qmatmul_node = onnx.helper.make_node(
        "QLinearMatMul",
        qmatmul_inputs,
        [match.node.output[0]],
        "{}_quant".format(match.node.name),
    )
    model.graph.node.append(qmatmul_node)
    return model


class ConvertQuantizableMatmul(OnnxTransform):
    """
    A transform that attempts, if possible, to convert MatMul nodes into
    their quantized representation.
    This MatMul is the result of quantizing native torch.matmul using QATMatMul

    | Starting with:
    |          INPUT_0           INPUT_1
    |            |               |
    |     QuantizeLinear     QuantizeLinear
    |            |               |
    |     DequantizeLinear   DequantizeLinear
    |                  |      |
    |                   MatMul
    |                     |
    |                  Transpose (optional)
    |                     |
    |                  Reshape (optional)
    |                     |
    |               QuantizeLinear
    |                     |
    |              DequantizeLinear
    |                     |
    |                  OUTPUT
    |
    | We end up converting to:
    |          INPUT_0           INPUT_1
    |            |               |
    |     QuantizeLinear     QuantizeLinear
    |                  |      |
    |                  |      |
    |                   QLinearMatMul
    |                     |
    |                   Transpose (optional)
    |                     |
    |                    Reshape (optional)
    |                     |
    |              DequantizeLinear
    |                     |
    |                  OUTPUT
    """

    def transform(self, model: ModelProto) -> ModelProto:
        graph = ONNXGraph(model)
        count_converted_nodes = 0
        for match in get_structural_matches(
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
        ):
            _LOGGER.debug(f"Matched quantizable MatMul: {match.node.name}")

            # Convert
            model = convert_matmul_to_quantized(match, model)
            count_converted_nodes += 1
            remove_node_and_params_from_graph(model, match.node)
            ONNXGraph(model).sort_nodes_topologically()

        if count_converted_nodes > 0:
            _LOGGER.info(
                f"Converted {count_converted_nodes} quantizable MatMul ops "
                "to QLinearMatMul"
            )
        return model
