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

from typing import List, Optional, Union

from onnx import NodeProto, TensorProto

from sparseml.onnx.utils import ONNXGraph


__all__ = [
    "INITIALIZER_MATCH",
    "optional_node",
    "MatchResult",
    "NodeOrInitializer",
    "get_structural_matches",
    "any_of",
]

INITIALIZER_MATCH = "__Initializer__"
"""
A special tag **ONLY** used for matching code to
match against initializers in an onnx graph
"""

_OPTIONAL_TAG = "Optional-"
_ANY_TAG = "Any-"


def optional_node(op_type: str) -> str:
    """
    Tells :func:`get_structural_matches` that this op type is an optional one.

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
    return _OPTIONAL_TAG + op_type


def any_of(*op_type: str) -> str:
    """
    Tells :func:`get_structural_matches` that this can be a set of op types

    ```python
    get_structural_matches(
        ...,
        children_ops=[
            [
                any_of("QuantizeLinear", "DequantizeLinear"),
            ]
        ]
    )
    ```
    """
    return _ANY_TAG + "-".join(op_type)


def get_structural_matches(
    graph: ONNXGraph,
    op_type: str,
    parent_ops: Optional[List[List[str]]] = None,
    children_ops: Optional[List[List[str]]] = None,
) -> List["MatchResult"]:
    """
    Gathers all nodes in the `graph` that match the `op_type` and the
    have the specified parent/children structure,
    controlled via parent_ops/children_ops.

    ### op_type example

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

    ### `parent_ops` and `INITIALIZER_MATCH` example

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

    ### Empty `parent_ops` example

    Another example is matching against a single parent branch of a node. In this
    case you can just specify `[]` as one of the parent ops:

    ```python
    matches = get_structural_matches(
        graph,
        op_type="Add",
        parent_ops=[
            [],
            ["QuantizeLinear", "DequantizeLinear"]
        ]
    )
    for match in matches:
        add_node = match.node
        assert len(match.parents[0]) == 0
        parent_1_quant, parent_1_dequant = match.parents[1]
    ```

    ### `optional_node` example

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

    ### `any_of` example

    Here is an example using the `any_of` function to match a parent node against
    a set of op_types:

    ```python
    matches = get_structural_matches(
        graph,
        op_type="MatMul",
        parent_ops=[[any_of("QuantizeLinear", "DequantizeLinear")]]
    )
    for match in matches:
        (quant_or_dequant, ) = match.parents[0]
        assert quant_or_dequant.op_type in ["QuantizeLinear", "DequantizeLinear"]
    ```


    :param graph: the graph to search in
    :param op_type: The `NodeProto.op_type` to match against
    :param parent_ops: List of List of `NodeProto.op_type` or `INITIALIZER_MATCH`
        to match against
    :param children_ops: List of List of `NodeProto.op_type` or `INITIALIZER_MATCH`
        to match against
    """

    # NOTE: gather matches completely first, so we don't have to worry about
    # updates to the graph messing with the iteration
    matches = []
    for node in graph._model.graph.node:
        match = _match_structure(graph, node, op_type, parent_ops, children_ops)
        if match is not None:
            matches.append(match)
    return matches


NodeOrInitializer = Union[NodeProto, TensorProto]
"""
Represents either:
1. a node (NodeProto) in an onnx graph
2. an initializer (TensorProto)
"""


class MatchResult:
    """
    The output of :func:`get_structural_matches`
    """

    def __init__(self, node: NodeOrInitializer):
        self.node: Optional[NodeOrInitializer] = node
        """
        The main node that was matched at top level. This node will have the op_type
        passed into the matching functions.
        """

        self.parents: List[List[Optional[NodeOrInitializer]]] = []
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

        self.children: List[List[Optional[NodeOrInitializer]]] = []
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

    def __str__(self) -> str:
        node_name = repr(self.node.name) if self.node is not None else None
        parent_names = [
            [p.name if p is not None else None for p in ps] for ps in self.parents
        ]
        children_names = [
            [c.name if c is not None else None for c in cs] for cs in self.children
        ]
        return (
            f"MatchResult(node={node_name}, "
            f"parents={parent_names}, "
            f"children={children_names})"
        )


def _match_structure(
    graph: ONNXGraph,
    node: Union[NodeProto, TensorProto],
    op_type: str,
    parent_ops: Optional[List[List[str]]] = None,
    children_ops: Optional[List[List[str]]] = None,
) -> Optional[MatchResult]:
    match = MatchResult(node)
    if not _match_op_type(match, graph, node, op_type):
        return None
    if parent_ops and not _match_parents(match, graph, node, parent_ops):
        return None
    if children_ops and not _match_children(match, graph, node, children_ops):
        return None
    return match


def _match_op_type(
    match: MatchResult, graph: ONNXGraph, node: NodeOrInitializer, op_type: str
) -> bool:
    if node is None:
        return False

    if op_type.startswith(_OPTIONAL_TAG):
        op_type = op_type.split("-")[1]
        is_optional = True
    else:
        is_optional = False

    if op_type.startswith(_ANY_TAG):
        op_types = op_type.split("-")[1:]
    else:
        op_types = [op_type]

    # match against initializers
    if node.name in graph._name_to_initializer and INITIALIZER_MATCH not in op_types:
        return False

    if isinstance(node, NodeProto) and node.op_type not in op_types:
        if is_optional:
            # NOTE: this is handled in `_match_children`
            match.node = None
        return is_optional

    return True


def _match_parents(
    match: MatchResult,
    graph: ONNXGraph,
    node: Union[NodeProto, TensorProto],
    parent_ops: List[List[str]],
) -> bool:
    if not (isinstance(node, NodeProto) and len(parent_ops) <= len(node.input)):
        return False

    parents = graph.get_node_parents(node)
    assert len(parents) == len(node.input)
    for parent, expected_op_sequence in zip(parents, parent_ops):
        # this case represents when a user passes in an `[]` for one of the elements
        # of parent_ops. In which case any parent in this slot is matched.
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
            return False
        match.parents.append((sub_match.parents[0] if head else []) + [sub_match.node])

    return True


def _match_children(
    match: MatchResult,
    graph: ONNXGraph,
    node: Union[NodeProto, TensorProto],
    children_ops: List[List[str]],
) -> bool:
    if not isinstance(node, NodeProto):
        return False

    children = graph.get_node_children(node)
    if children == [] and all(
        all(op.startswith(_OPTIONAL_TAG) for op in ops) for ops in children_ops
    ):
        # we are at the end of the graph and all children nodes are optional
        # this is considered a match
        for ops in children_ops:
            match.children.append([None for _ in ops])
        return True

    # NOTE: get_node_children can return less than node.output if one of the outputs
    #       is the graph output. this is a difference in behavior to
    #       get_node_parents, which replaces input with None, instead of removing it.
    # NOTE: comparing to length of children here also handles the case where a node
    #       has a single output that is used by multiple children nodes
    if len(children_ops) > len(children):
        return False
    for child, expected_op_sequence in zip(children, children_ops):
        # this case represents when a user passes in an `[]` for one of the elements
        # of children_ops. In which case any child in this slot is matched.
        if expected_op_sequence == []:
            match.children.append([])
            continue

        head, *tail = expected_op_sequence

        # NOTE: explicitly only matching against a single child output here
        #       even though it could have multiple outputs
        sub_match = _match_structure(
            graph,
            # NOTE: here's where we handle optional children via the match.node is None
            #       if we are on an optional node that didn't match, we keep using
            #       `node` instead of recursing on `child`.
            node=node if match.node is None else child,
            op_type=head,
            children_ops=[tail] if tail else None,
        )
        if sub_match is None:
            return False
        match.children.append(
            [sub_match.node] + (sub_match.children[0] if tail else [])
        )

        # sanity checks
        assert len(match.children[-1]) == len(expected_op_sequence)

    return True
