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

from typing import Iterable, List, NamedTuple, Optional, Set, Tuple, Union

import numpy
from onnx import ModelProto, NodeProto, TensorProto, numpy_helper

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


NodeOrInit = Union[NodeProto, TensorProto]


class MatchResult:
    def __init__(self, node: NodeOrInit) -> None:
        self.node: Optional[NodeOrInit] = node
        self.parents: List[Optional[NodeOrInit]] = []
        self.children: List[Optional[NodeOrInit]] = []


INITIALIZER_MATCH = "__Initializer__"


def optional_node(tag: str) -> str:
    return "Optional-" + tag


def _is_optional_node(tag: str) -> Tuple[str, bool]:
    if tag.startswith("Optional-"):
        return tag.split("-")[1], True
    else:
        return tag, False


def iter_structural_matches(
    graph: ONNXGraph,
    op_type: Optional[str] = None,
    parent_ops: Optional[List[List[str]]] = None,
    children_ops: Optional[List[List[str]]] = None,
) -> Iterable[MatchResult]:
    for node in graph._model.graph.node:
        match = match_structure(
            graph,
            node,
            op_type=op_type,
            parent_ops=parent_ops,
            children_ops=children_ops,
        )
        if match is not None:
            yield match


def match_structure(
    graph: ONNXGraph,
    node: Union[NodeProto, TensorProto],
    op_type: Optional[str] = None,
    parent_ops: Optional[List[List[str]]] = None,
    children_ops: Optional[List[List[str]]] = None,
) -> Optional[MatchResult]:
    """ """
    match = MatchResult(node)

    if op_type is not None:
        op_type, is_optional = _is_optional_node(op_type)
        if is_optional and op_type == INITIALIZER_MATCH:
            raise NotImplementedError("optional initializers not supported")

        if op_type == INITIALIZER_MATCH and not isinstance(node, TensorProto):
            return None
        if op_type != INITIALIZER_MATCH and not (
            isinstance(node, NodeProto) and node.op_type == op_type
        ):
            if is_optional:
                match.node = None
            else:
                return None

    if parent_ops:
        if not (isinstance(node, NodeProto) and len(parent_ops) == len(node.input)):
            return None

        parents = graph.get_node_parents(node)
        assert len(parents) == len(parent_ops)
        for p, ops in zip(parents, parent_ops):
            sub_match = match_structure(
                graph,
                p,
                op_type=ops[-1],
                parent_ops=[ops[:-1]] if len(ops) > 1 else None,
            )
            if sub_match is None:
                return None
            match.parents.append([sub_match.node] + sub_match.parents)

    if children_ops:
        if not (isinstance(node, NodeProto) and len(children_ops) == len(node.output)):
            return None

        children = graph.get_node_children(node)
        assert len(children) == len(children_ops)
        for c, ops in zip(children, children_ops):
            sub_match = match_structure(
                graph,
                c,
                op_type=ops[-1],
                children_ops=[ops[:-1]] if len(ops) > 1 else None,
            )
            if sub_match is None:
                return None
            match.children.append([sub_match.node] + sub_match.children)

    return match
