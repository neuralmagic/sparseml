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

from onnx import ModelProto, NodeProto
from typing import List

from sparseml.onnx.utils import remove_node_and_params_from_graph


_QUANTIZE_OP_NAMES = ["QuantizeLinear", "DequantizeLinear"]


def delete_quant_node(
    model: ModelProto,
    node: NodeProto,
    keep_weight: bool = False,
):
    """
    Deletes a QuantizeLinear or DequantizeLinear and its parameters from the model
    :param model: ONNX model to modify
    :param node: the QuantizeLinear or DequantizeLinear node to delete
    :param keep_weight: set true to not delete the weight param possibly stored as an
        initializer to the first input of this node
    """
    assert (
        node.op_type in _QUANTIZE_OP_NAMES
    ), "Op Type must be either QuantizeLinear or DequantizeLinear, found {} ".format(
        node.op_type
    )
    if keep_weight:
        del node.input[0]
    remove_node_and_params_from_graph(model, node)


def check_for_sequence_of_children_nodes(node: NodeProto, graph: "ONNXGraph", node_sequence: List[str]) -> bool:
    """
    Checks if a sequence of nodes appears after the given node.
    Essentially performs a depth-first search starting from the given node.

    :param node: the node to check
    :param model: the model to check
    :param node_sequence: the sequence of nodes to check for
    :return: True if the sequence of nodes follows the given node, False otherwise
    """
    for expected_node in node_sequence:
        child_nodes = graph.get_node_children(node)
        for child_node in child_nodes:
            if check_node_op(child_node, expected_node):
                node = child_node
                break
            return False
    return True

def check_for_sequence_of_parent_nodes(node: NodeProto, graph: "ONNXGraph", node_sequence: List[str]) -> bool:
    """
    Checks if a sequence of nodes appears before the given node.
    Essentially performs a depth-first search starting from the given node
    (backwards -> towards the root of the tree).

    :param node: the node to check
    :param model: the model to check
    :param node_sequence: the sequence of nodes to check for
    :return: True if the sequence of nodes precedes the given node, False otherwise
    """
    for expected_node in node_sequence:
        parent_nodes = graph.get_node_parents(node)
        for parent_node in parent_nodes:
            if check_node_op(parent_node, expected_node):
                node = parent_node
                break
            return False
    return True

def check_node_op(node: NodeProto, op: str) -> bool:
    """
    Checks if a node has the given op type

    :param node: the node to check
    :param op: the op type to check for
    :return: True if the node has the given op type, False otherwise
    """
    if node is None:
        return False
    return node.op_type == op