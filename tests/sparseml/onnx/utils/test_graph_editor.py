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

from typing import List

import numpy
import onnx
import pytest
from onnx import load_model

from sparseml.onnx.utils import (
    ONNXGraph,
    get_node_params,
    prune_model_one_shot,
    prune_unstructured,
)
from tests.sparseml.onnx.helpers import OnnxRepoModelFixture


from tests.sparseml.onnx.helpers import onnx_repo_models  # noqa isort: skip


def _test_correct_sparsity(pruned_array, sparsity, tolerance=1e-4):
    measured_sparsity = float(
        pruned_array.size - numpy.count_nonzero(pruned_array)
    ) / float(pruned_array.size)
    assert abs(measured_sparsity - sparsity) < tolerance


def _test_correct_pruning(pruned_array, sparse_threshold):
    pruned_correctly = numpy.vectorize(
        lambda x: True if abs(x) > abs(sparse_threshold) or x == 0 else x
    )
    assert pruned_correctly(pruned_array).all()


@pytest.mark.parametrize(
    "array,sparsities",
    [
        (
            numpy.random.randn(3, 128, 128),
            [0.01, 0.1, 0.25, 0.5, 0.8, 0.9, 0.99, 0.999],
        ),
        (
            numpy.random.randn(3, 128, 128) - 1,
            [0.01, 0.1, 0.25, 0.5, 0.8, 0.9, 0.99, 0.999],
        ),
        (
            numpy.random.randn(3, 128, 128) + 1,
            [0.01, 0.1, 0.25, 0.5, 0.8, 0.9, 0.99, 0.999],
        ),
    ],
)
def test_prune_unstructured(array, sparsities):
    sorted_array = numpy.sort(numpy.abs(array.flatten()))

    for sparsity in sparsities:
        sparse_index = round(sparsity * array.size) - 1
        sparse_threshold = sorted_array[sparse_index]

        pruned_array = prune_unstructured(array, sparsity)
        _test_correct_sparsity(pruned_array, sparsity, 1e-4)
        _test_correct_pruning(pruned_array, sparse_threshold)


@pytest.mark.parametrize("sparsity", [(0.01), (0.5), (0.99), (0.999)])
def test_prune_model_one_shot(
    onnx_repo_models: OnnxRepoModelFixture, sparsity: List[float]  # noqa: F811
):
    model_path = onnx_repo_models.model_path
    model = load_model(model_path)
    nodes = [
        node
        for node in model.graph.node
        if node.op_type == "Conv" or node.op_type == "Gemm"
    ]
    prune_model_one_shot(model, nodes, sparsity)

    for node in nodes:
        weight, _ = get_node_params(model, node)
        _test_correct_sparsity(weight.val, sparsity, 5.5e-3)


def test_prune_model_one_shot_sparsity_list(
    onnx_repo_models: OnnxRepoModelFixture,  # noqa: F811
):
    model_path = onnx_repo_models.model_path
    model = load_model(model_path)
    nodes = [
        node
        for node in model.graph.node
        if node.op_type == "Conv" or node.op_type == "Gemm"
    ]
    sparsities = numpy.random.random_sample([len(nodes)])

    prune_model_one_shot(model, nodes, sparsities)

    for node, sparsity in zip(nodes, sparsities):
        weight, _ = get_node_params(model, node)
        weight_val = weight.val
        _test_correct_sparsity(weight_val, sparsity, 5.5e-3)


def test_sort_nodes_topologically(
    onnx_repo_models: OnnxRepoModelFixture,  # noqa: F811
):
    model_path = onnx_repo_models.model_path
    model = load_model(model_path)
    onnx.checker.check_model(model)  # assert that starting model is valid

    # create invalid model by changing node ordering
    nodes = list(model.graph.node)
    checker_failed = False
    for _ in range(10):
        try:
            numpy.random.shuffle(nodes)
            model.graph.ClearField("node")
            model.graph.node.extend(nodes)
            onnx.checker.check_model(model)
        except Exception:
            # checker failed due to node topo ordering
            checker_failed = True
            break
    assert checker_failed, "unable to induce invalid model by shuffling node order"

    graph = ONNXGraph(model)
    graph.sort_nodes_topologically()
    # check that sorted model is valid
    onnx.checker.check_model(model)
