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
from typing import Optional, Tuple, Union

import onnx
from onnx import ModelProto, NodeProto

from sparseml.exporters.transforms import BaseTransform
from sparseml.exporters.transforms.helpers import delete_quant_node
from sparseml.onnx.utils import (
    ONNXGraph,
    check_load_model,
    remove_node_and_params_from_graph,
    validate_onnx_file,
)


_LOGGER = logging.getLogger(__name__)

OPTIONAL_NODES_NAMES = {"Transpose", "Reshape"}


def is_node_quantize_linear(node: NodeProto) -> bool:
    """
    Checks if a node is a valid quantize linear node
    """
    if node is None:
        return False
    return node.op_type == "QuantizeLinear"


def is_node_dequantize_linear(node: NodeProto) -> bool:
    """
    Checks if a node is a valid dequantize linear node
    """
    if node is None:
        return False
    return node.op_type == "DequantizeLinear"


def dequant_quant_previous(matmul_node: NodeProto, graph: "ONNXGraph") -> bool:
    """
    Checks whether the inputs to the matmul_node are composed of two nodes:
    - dequantize_linear
    - quantize_linear nodes.
    In other words:
    - both parent nodes of the matmul_node are `dequantize_linear` nodes
    - both grandparent nodes (each parent has only a single parent) of
        the `matmul_node` are `quantize_linear` nodes

    :param matmul_node: A matmul node
        (i.e a node with op_type == "MatMul")
    :param graph: An ONNX graph that the node belongs to
    :return: a boolean flag indicating whether the matmul_node fulfills
        the conditions described above
    """
    parent_nodes = [graph.get_node_single_parent(matmul_node, i) for i in range(2)]
    for parent_node in parent_nodes:
        if not is_node_dequantize_linear(parent_node):
            return False
        grandparent_node = graph.get_node_single_parent(parent_node, 0)
        if not is_node_quantize_linear(grandparent_node):
            return False
    return True


def quant_dequant_next(node: NodeProto, graph: "ONNXGraph") -> bool:
    """
    Checks whether the node is followed by a quantize_linear node
    and then by a dequantize_linear node.

    :param node: A node
    :param graph: An ONNX graph that the node belongs to
    :return: a boolean flag indicating whether the node fulfills
        the conditions described above
    """
    child_node = graph.get_node_single_child(node)

    if not is_node_quantize_linear(child_node):
        return False
    grandchild_node = graph.get_node_single_child(child_node)
    if not is_node_dequantize_linear(grandchild_node):
        return False
    return True


def check_optional_nodes(node: NodeProto, graph: "ONNXGraph") -> Optional[NodeProto]:
    """
    Checks whether the node is followed by at most two optional nodes

    :param node: A node
    :param graph: An ONNX graph that the node belongs to
    :return: a last optional node
    """
    child_node = graph.get_node_single_child(node)
    op_child_node = child_node.op_type
    if op_child_node not in OPTIONAL_NODES_NAMES:
        # no optional nodes detected
        return None
    grandchild_node = graph.get_node_single_child(child_node)
    op_grandchild_node = grandchild_node.op_type
    if op_grandchild_node not in OPTIONAL_NODES_NAMES:
        # one optional node detected
        return child_node
    # two optional nodes detected; returning the last one
    return grandchild_node


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

    if not dequant_quant_previous(matmul_node, graph):
        # checking whether nodes prior to matmul are
        # dequantize_linear and quantize_linear
        return None
    # check whether matmul node is followed by any of the optional nodes
    last_node_optional = check_optional_nodes(matmul_node, graph)
    node = last_node_optional or matmul_node

    if not quant_dequant_next(node, graph):
        # checking whether nodes that come after
        # matmul (or optional nodes if present) are
        # quantize_linear and dequantize_linear
        return None
    output_quantize_node = graph.get_node_single_child(node)

    return output_quantize_node, last_node_optional


def convert_matmul_to_quantized(
    model: ModelProto,
    matmul_node: NodeProto,
    matmul_node_index: int,
    output_quantize_node: NodeProto,
    last_node_optional: Optional[NodeProto] = None,
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
    graph = ONNXGraph(model)
    # fetch two `DequantizeLinear` nodes that precede the `MatMul` node
    node_0, node_1 = [graph.get_node_single_parent(matmul_node, i) for i in range(2)]
    # construct inputs for the new `QLinearMatmul` node
    qmatmul_inputs = [
        node_0.input[0],  # a
        node_0.input[1],  # a_scale
        node_0.input[2],  # a_zero_point
        node_1.input[0],  # b
        node_1.input[1],  # b_scale
        node_1.input[2],  # b_zero_point
        output_quantize_node.input[1],  # y_scale
        output_quantize_node.input[2],  # y_zero_point
    ]

    # remove the `DequantizeLinear` nodes that precede the `MatMul` node,
    # as well as the `QuantizeLinear` node that follows the `MatMul` node
    # (or follows the last optional node if optional nodes present)
    for node_to_delete in [node_0, node_1, output_quantize_node]:
        delete_quant_node(model, node_to_delete)

    # delete original MatMul node
    remove_node_and_params_from_graph(model, matmul_node)

    # if optional nodes present, adjust the `DequantizeLinear` node that
    # follows the last optional node
    if last_node_optional:
        output_dequantize_node = graph.get_node_single_child(output_quantize_node)
        graph.update_node_input(
            node=output_dequantize_node,
            input_idx=0,
            input_id=last_node_optional.input[0],
        )

    # create qmatmul node and add it to graph
    qmatmul_output = (
        matmul_node.output[0] if last_node_optional else output_quantize_node.output[0]
    )
    qmatmul_name = "{}_quant".format(matmul_node.name)
    qmatmul_node = onnx.helper.make_node(
        "QLinearMatMul",
        qmatmul_inputs,
        [qmatmul_output],
        qmatmul_name,
    )
    matmul_node_index -= (
        2  # adjust index to account for two deleted nodes (node_0 and node_1)
    )

    model.graph.node.insert(matmul_node_index, qmatmul_node)
    return model


class ConvertQuantizableMatmul(BaseTransform):
    """
    A transform that attempts, if appropriate, to convert MatMul nodes into
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
        matmul_nodes_indices, matmul_nodes = zip(
            *[
                (idx, node)
                for idx, node in enumerate(model.graph.node)
                if node.op_type == "MatMul"
            ]
        )
        for idx, matmul_node in zip(matmul_nodes_indices, matmul_nodes):
            # Check whether there is a match
            result = is_quantizable_matmul(matmul_node, graph)
            if result is None:
                continue
            output_quantize_node, last_node_optional = result
            _LOGGER.debug(f"Matched quantizable MatMul: {matmul_node.name}")

            # Convert
            model = convert_matmul_to_quantized(
                model, matmul_node, idx, output_quantize_node, last_node_optional
            )
            count_converted_nodes += 1

        if matmul_nodes:
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
