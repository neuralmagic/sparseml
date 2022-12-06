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

from typing import List, NamedTuple, Optional, Set, Tuple, Union

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
"""
Represents either:
1. a node (NodeProto) in an onnx graph
2. an initializer (TensorProto)
"""


class MatchResult:
    """
    The output of :func:`get_structural_matches` and :func:`match_structure`.
    """

    def __init__(self, node: NodeOrInit) -> None:
        self.node: Optional[NodeOrInit] = node
        """
        The main node that was matched at top level. This node will have the op_type
        passed into the matching functions.
        """

        self.parents: List[List[Optional[NodeOrInit]]] = []
        """
        This is the sequence of parent nodes that was matched via the `parent_ops`
        keyword.

        For example, if you pass in:
        ```python
        match = get_structural_matches(..., parent_ops=[
            ["A", "B"],
            ["1", "2"],
            ["q", "w"],
            ["G", "H"],
        ])
        ```
        And a match is returned, this list be ordered and sized exactly
        the way the parent_ops list is ordered.
        i.e. you can destructure it like so
        ```python
        a, b = match.parents[0]
        _1, _2 = match.parents[1]
        q, w = match.parents[2]
        g, h = match.parents[3]
        ```
        """

        self.children: List[List[Optional[NodeOrInit]]] = []
        """
        This is the sequence of children nodes that were matched via the `children_ops`
        keyword.

        For example, if you pass in:
        ```python
        match = get_structural_matches(..., children_ops=[
            ["A", "B"],
            ["1", "2"],
            ["q", "w"],
            ["G", "H"],
        ])
        ```
        And a match is returned, this list be ordered and sized exactly
        the way the parent_ops list is ordered.
        i.e. you can destructure it like so
        ```python
        a, b = match.children[0]
        _1, _2 = match.children[1]
        q, w = match.children[2]
        g, h = match.children[3]
        ```
        """


INITIALIZER_MATCH = "__Initializer__"
"""
A special tag **ONLY** used for matching code to
match against initializers in an onnx graph
"""


def optional_node(op_type: str) -> str:
    """
    Tells :func:`get_structural_matches` and :func:`match_structure`
    that this op type is an optional one.

    e.g.
    ```python
    get_structural_matches(
        ...,
        children_ops=[
            [
                optional_node("Transpose")
                "QuantizeLinear",
            ]
        ]
    )
    ```
    """
    return "Optional-" + op_type


def _is_optional_node(tag: str) -> Tuple[str, bool]:
    if tag.startswith("Optional-"):
        return tag.split("-")[1], True
    else:
        return tag, False


def get_structural_matches(
    graph: ONNXGraph,
    op_type: str,
    parent_ops: Optional[List[List[str]]] = None,
    children_ops: Optional[List[List[str]]] = None,
) -> List[MatchResult]:
    """
    Gathers all nodes in the `graph` that match the `op_type` and the
    have the specified parent/children structure,
    controlled via parent_ops/children_ops.

    A simple example just matching against op_type:

        ```python
        matches = get_structural_matches(graph, op_type="Identity")
        for match in matches:
            id_node = match.node
            assert isinstance(id_node, onnx.NodeProto) and id_node.op_type == "Identity"
            assert match.parents == []
            assert match.children == []
        ```

    `parent_ops` and `children_ops` are list of list of op type strings.
    It's a list of lists because nodes can have multiple inputs and multiple outputs.
    So each sub list in `parent_ops` and `children_ops` will be
    matched against the corresponding entry in node.input/node.output. I.e.
        - parent_ops[0] will be compared against node.input[0]
        - parent_ops[2] will be compared against node.input[2]
        - children_ops[3] will be compared against node.output[3]
    Because of this, if the length of node.input/node.output is shorter
    than parent_ops/children_ops, it is not a possible match and will
    be discarded.

    Here's a simple example of matching against an identity node
    with a single parent (because there is a single list in parent_ops),
    and the parent must be an initializer (the INITIALIZER_MATCH value):
    ```python
    matches = get_structural_matches(
        graph,
        op_type="Identity",
        parent_ops=[[INITIALIZER_MATCH]]
    )
    for match in matches:
        id_node = match.node
        (init, ) = match.parents[0]
        assert isinstance(init, onnx.TensorProto)
    ```

    Here is a really complicated example with optional nodes and multiple parents. Here
    we match against MatMul nodes that have at least 2 inputs from Quant/Dequant
    sequences. After the MatMul node there are two optional nodes followed by
    a Quant/Dequant sequence.

    If the optional_nodes are found, then the values in children_ops will be the nodes.
    If the optional nodes are NOT found, then those entries in children_ops
    will be None.

    In both cases, the length of match.children[0] will be the same as
    the length of children_ops[0].

    ```python
    matches = get_structural_matches(
        graph,
        op_type="MatMul",
        parent_ops=[
            ["QuantizeLinear", "DequantizeLinear"],
            ["QuantizeLinear", "DequantizeLinear"],
        ],
        children_ops=[
            [
                optional_node("Transpose"),
                optional_node("Reshape"),
                "QuantizeLinear",
                "DequantizeLinear",
            ]
        ]
    )
    for match in matches:
        matmul_node = match.node
        parent_0_quant, parent_0_dequant = match.parents[0]
        parent_1_quant, parent_1_dequant = match.parents[1]
        opt_transpose, opt_reshape, child_quant, child_dequant = match.children[0]
    ```
    """

    # NOTE: gather matches completely first, so we don't have to worry about
    # updates to the graph messing with the iteration
    matches = []
    for node in graph._model.graph.node:
        match = _match_structure(graph, node, op_type, parent_ops, children_ops)
        if match is not None:
            matches.append(match)
    return matches


def _match_structure(
    graph: ONNXGraph,
    node: Union[NodeProto, TensorProto],
    op_type: str,
    parent_ops: Optional[List[List[str]]] = None,
    children_ops: Optional[List[List[str]]] = None,
) -> Optional[MatchResult]:
    match = MatchResult(node)

    op_type, is_optional = _is_optional_node(op_type)

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
        if not (isinstance(node, NodeProto) and len(parent_ops) <= len(node.input)):
            return None

        parents = graph.get_node_parents(node)
        assert len(parents) == len(node.input)
        for parent, expected_op_sequence in zip(parents, parent_ops):
            if expected_op_sequence == []:
                match.parents.append([])
                continue

            # NOTE: since these are before the current parent, we iterate backwards
            *head, tail = expected_op_sequence

            # NOTE: explicitly only matching against a single parent input here
            #       even though it could have multiple inputs
            sub_match = _match_structure(
                graph, node=parent, op_type=tail, parent_ops=[head] if head else None
            )
            if sub_match is None:
                return None
            match.parents.append(
                (sub_match.parents[0] if head else []) + [sub_match.node]
            )

    if children_ops:
        if not (isinstance(node, NodeProto) and len(children_ops) <= len(node.output)):
            return None

        children = graph.get_node_children(node)
        # NOTE: get_node_children can return less than node.output if one of the outputs
        #       is the graph output.
        # this is a difference in behavior to get_node_parents, which replaces input
        # with None, instead of removing it.
        if not (len(children_ops) <= len(children)):
            return None
        for child, expected_op_sequence in zip(children, children_ops):
            if expected_op_sequence == []:
                match.children.append([])
                continue

            head, *tail = expected_op_sequence

            # NOTE: explicitly only matching against a single child output here
            #       even though it could have multiple outputs
            sub_match = _match_structure(
                graph,
                node=node if match.node is None else child,
                op_type=head,
                children_ops=[tail] if tail else None,
            )
            if sub_match is None:
                return None
            match.children.append(
                [sub_match.node] + (sub_match.children[0] if tail else [])
            )

            # sanity checks
            assert len(match.children[-1]) == len(expected_op_sequence)
    return match
