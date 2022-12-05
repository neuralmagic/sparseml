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
from typing import Optional, Set, Tuple, Union

import onnx
from onnx import ModelProto, NodeProto

from sparseml.exporters.transforms import BaseTransform
from sparseml.exporters.transforms.utils import (
    assert_node_type,
    check_for_sequence_of_children_nodes,
    check_for_sequence_of_parent_nodes,
    delete_quant_node,
    get_structural_matches,
    MatchResult,
    optional_node,
)
from sparseml.onnx.utils import (
    ONNXGraph,
    check_load_model,
    remove_node_and_params_from_graph,
    validate_onnx_file,
)


_LOGGER = logging.getLogger(__name__)

OPTIONAL_NODES_NAMES = {"Transpose", "Reshape"}


def check_optional_nodes(
    node: NodeProto,
    graph: "ONNXGraph",
    optional_nodes_names: Set[str] = OPTIONAL_NODES_NAMES,
) -> Optional[NodeProto]:
    """
    Checks whether the node is followed by a sequence of optional nodes.
    Assume a linear sequence of optional nodes.

    :param node: A node
    :param graph: An ONNX graph that the node belongs to
    :return: a last optional node if available, None otherwise
    """
    last_optional_node = None
    while True:
        child_node = graph.get_node_single_child(node)
        if child_node.op_type not in optional_nodes_names:
            break
        node = child_node
        last_optional_node = child_node

    return last_optional_node


def is_quantizable_matmul(
    matmul_node: NodeProto, graph: "ONNXGraph"
) -> Optional[Tuple[NodeProto, NodeProto]]:
    """
    Checks if a matmul node is quantizable

    :param matmul_node: A matmul node (i.e a node with op_type == "MatMul")
    :param graph: An ONNX graph that the node belongs to
    :return: One of the following:
        - None if the matmul node is not quantizable
        - A tuple of the
            1. quantizable matmul node and
            2. last optional node (if any optional nodes present in the graph)
        - A tuple of the
            1. quantizable matmul node and
            2. None (if no optional nodes present in the graph)
    """
    parent_nodes = [graph.get_node_single_parent(matmul_node, i) for i in range(2)]
    for parent_node in parent_nodes:
        # assert parent nodes of MatMul are DequantizeLinear
        if not assert_node_type(parent_node, "DequantizeLinear"):
            return None
        # assert that grandparent nodes of MatMul are QuantizeLinear
        if not check_for_sequence_of_parent_nodes(
            node=parent_node, graph=graph, node_sequence=["QuantizeLinear"]
        ):
            return None

    # check whether matmul node is followed by any of the optional nodes
    last_node_optional = check_optional_nodes(matmul_node, graph)
    node = last_node_optional or matmul_node

    if not check_for_sequence_of_children_nodes(
        node=node, graph=graph, node_sequence=["QuantizeLinear", "DequantizeLinear"]
    ):
        return None

    output_quantize_node = graph.get_node_single_child(node)
    return output_quantize_node, last_node_optional


def convert_matmul_to_quantized(
    model: ModelProto,
    match: MatchResult,
) -> ModelProto:
    """
    Converts a matmul node to a quantized matmul node
    and updates the onnx model accordingly

    :param model: The ONNX model that the matmul node belongs to
    :param matmul_node: A matmul node that will be converted to a
        quantized matmul node
    :param matmul_node_index: The index of the matmul node in the model
    :param output_quantize_node: The quantize_linear node that follows
        the matmul node (or follows the `last_node_optional` if not None)
    :param last_node_optional: An optional node that precedes the `output_quantize_node`
    :return: The modified ONNX model
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

    # delete original MatMul node
    remove_node_and_params_from_graph(model, match.node)

    # set dequantize's input to quant's input
    # NOTE: this handles the prescence of the optional transpose/reshape nodes
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


class ConvertQuantizableMatmul(BaseTransform):
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

    def _transform(self, model: ModelProto) -> ModelProto:
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
            model = convert_matmul_to_quantized(model, match)
            count_converted_nodes += 1

        ONNXGraph(model).sort_nodes_topologically()
        if count_converted_nodes > 0:
            _LOGGER.info(
                f"Converted {count_converted_nodes} quantizable MatMul ops "
                "to QLinearMatMul"
            )
        return model

    def _validate_input(self, model: ModelProto):
        validate_onnx_file(model)

    def _validate_output(self, model: ModelProto):
        validate_onnx_file(model)

    def apply(self, model: Union[ModelProto, str]) -> ModelProto:
        onnx_model = check_load_model(model)
        self._validate_input(onnx_model)
        onnx_model = self._transform(onnx_model)
        self._validate_output(onnx_model)
        return onnx_model
