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

import onnx
import pytest

from sparseml.exporters.transforms.utils.matching import (
    INITIALIZER_MATCH,
    any_of,
    get_structural_matches,
    optional_node,
)
from sparseml.onnx.utils.graph_editor import ONNXGraph


@pytest.fixture()
def onnx_graph() -> ONNXGraph:
    """
    Creates a graph that looks like:

    init1
     |
    id1     input
      |       |
       \\    /
         add1   init2
          |    /
          |   /
          add2
           |
        output
    """
    model_input = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, (1,)
    )
    model_output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, (1,)
    )
    init1 = onnx.helper.make_tensor(
        name="init1", data_type=onnx.TensorProto.FLOAT, dims=(1,), vals=[1]
    )
    init2 = onnx.helper.make_tensor(
        name="init2", data_type=onnx.TensorProto.FLOAT, dims=(1,), vals=[2]
    )
    id1 = onnx.helper.make_node("Identity", ["init1"], ["id1_output"], name="id1")
    add1 = onnx.helper.make_node(
        "Add", ["id1_output", "input"], ["add1_output"], name="add1"
    )
    add2 = onnx.helper.make_node(
        "Add", ["add1_output", "init2"], ["add2_output"], name="add2"
    )
    id2 = onnx.helper.make_node("Identity", ["add2_output"], ["output"], name="id2")
    graph = onnx.helper.make_graph(
        nodes=[id1, add1, add2, id2],
        name="g",
        inputs=[model_input],
        outputs=[model_output],
        initializer=[init1, init2],
    )

    model = onnx.helper.make_model(graph)
    return ONNXGraph(model)


def test_match_optional_nodes(onnx_graph: ONNXGraph):
    matches = get_structural_matches(
        onnx_graph, op_type="Add", children_ops=[[optional_node("Add")]]
    )
    assert list(map(str, matches)) == [
        "MatchResult(node='add1', parents=[], children=[['add2']])",
        "MatchResult(node='add2', parents=[], children=[[None]])",
    ]


def test_match_all_options(onnx_graph: ONNXGraph):
    matches = get_structural_matches(
        onnx_graph,
        parent_ops=[
            ["Add"],
            [INITIALIZER_MATCH],
        ],
        op_type="Add",
        children_ops=[["Identity"]],
    )
    assert list(map(str, matches)) == [
        "MatchResult(node='add2', parents=[['add1'], ['init2']], children=[['id2']])"
    ]


def test_only_op_type(onnx_graph: ONNXGraph):
    matches = get_structural_matches(onnx_graph, op_type="Conv")
    assert len(matches) == 0

    matches = get_structural_matches(onnx_graph, op_type="Identity")
    assert list(map(str, matches)) == [
        "MatchResult(node='id1', parents=[], children=[])",
        "MatchResult(node='id2', parents=[], children=[])",
    ]

    matches = get_structural_matches(onnx_graph, op_type="Add")
    assert list(map(str, matches)) == [
        "MatchResult(node='add1', parents=[], children=[])",
        "MatchResult(node='add2', parents=[], children=[])",
    ]


def test_parent_empty(onnx_graph: ONNXGraph):
    matches = get_structural_matches(
        onnx_graph,
        op_type="Add",
        parent_ops=[
            [],
            [INITIALIZER_MATCH],
        ],
    )
    assert list(map(str, matches)) == [
        "MatchResult(node='add2', parents=[[], ['init2']], children=[])"
    ]


def test_match_any(onnx_graph: ONNXGraph):
    matches = get_structural_matches(
        onnx_graph,
        op_type="Add",
        parent_ops=[
            [any_of("Add", "Identity")],
            [],
        ],
    )

    assert list(map(str, matches)) == [
        "MatchResult(node='add1', parents=[['id1'], []], children=[])",
        "MatchResult(node='add2', parents=[['add1'], []], children=[])",
    ]
