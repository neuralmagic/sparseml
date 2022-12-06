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

from typing import List, NamedTuple, Set, Union

import numpy
from onnx import ModelProto, NodeProto, numpy_helper

from sparseml.onnx.utils import ONNXGraph, remove_node_and_params_from_graph


_QUANTIZE_OP_NAMES = ["QuantizeLinear", "DequantizeLinear"]

"""
Named tuple object to represent scale/zero point values for quantizing tenors
"""

QuantizationParams = NamedTuple(
    "QuantizationParams",
    [("scale", float), ("zero_point", int), ("target", Union[numpy.ndarray, None])],
)


def get_quantization_params(
    model: Union[ModelProto, ONNXGraph],
    node: NodeProto,
    include_target: bool = False,
) -> QuantizationParams:
    """
    :param model: ONNX model to read from or ONNXGraph object
    :param node: A QuantizeLinear or DequantizeLinear Node
    :param include_target: Set True include quantization target. If False,
        target value will be returned as None. Default is None
    :return: QuantizationParams object with scale and zero point, will include the
         quantization target if it is an initializer otherwise target will be None
    """
    assert (
        node.op_type in _QUANTIZE_OP_NAMES
    ), "Op Type must be either QuantizeLinear or DequantizeLinear, found {} ".format(
        node.op_type
    )

    graph = model if isinstance(model, ONNXGraph) else ONNXGraph(model)

    scale = graph.get_init_by_name(node.input[1])
    if scale is None:
        scale_const = graph.get_node_by_output_id(node.input[1])
        if scale_const:
            scale = scale_const.attribute[0].t
    assert scale, "Quantization scale {} not found".format(node.input[1])

    zero_point = graph.get_init_by_name(node.input[2])
    if zero_point is None:
        zero_point_const = graph.get_node_by_output_id(node.input[2])
        if zero_point_const:
            zero_point = zero_point_const.attribute[0].t
    assert zero_point, "Quantization zero point {} not found".format(node.input[2])

    scale = numpy_helper.to_array(scale)
    zero_point = numpy_helper.to_array(zero_point)

    target = None
    if include_target:
        target = graph.get_init_by_name(node.input[0])
        if target is not None:
            target = numpy_helper.to_array(target)

    return QuantizationParams(scale=scale, zero_point=zero_point, target=target)


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


def check_for_sequence_of_children_nodes(
    node: NodeProto, graph: "ONNXGraph", node_sequence: List[str]
) -> bool:
    """
    Checks if a sequence of nodes appears after the given node.
    It does so by performing a depth-first search starting from the given node.
    (forward -> towards the leaves of the tree).

    :param node: the node to check
    :param model: the model to check
    :param node_sequence: the sequence of nodes to check for
    :return: True if the sequence of nodes follows the given node, False otherwise
    """
    for expected_node in node_sequence:
        child_nodes = graph.get_node_children(node)
        for child_node in child_nodes:
            if assert_node_type(child_node, expected_node):
                node = child_node
                break
            return False
    return True


def check_for_sequence_of_parent_nodes(
    node: NodeProto, graph: "ONNXGraph", node_sequence: List[str]
) -> bool:
    """
    Checks if a sequence of nodes appears before the given node.
    It does so by performing a depth-first search starting from the given node
    (backwards -> towards the root of the tree).

    :param node: the node to check
    :param model: the model to check
    :param node_sequence: the sequence of nodes to check for
    :return: True if the sequence of nodes precedes the given node, False otherwise
    """
    for expected_node in node_sequence:
        parent_nodes = graph.get_node_parents(node)
        for parent_node in parent_nodes:
            if assert_node_type(parent_node, expected_node):
                node = parent_node
                break
            return False
    return True


def assert_node_type(node: NodeProto, op: Union[List[str], Set[str], str]) -> bool:
    """
    Checks if a node is of the given op type

    :param node: the node to check
    :param op: the operation type to check for
    :return: True if the node has the given op type, False otherwise
    """
    if node is None:
        return False
    if isinstance(op, str):
        return node.op_type == op
    else:
        return node.op_type in op
