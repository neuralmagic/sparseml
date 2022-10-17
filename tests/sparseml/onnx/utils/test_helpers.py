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

import json
import os

import numpy
import pytest
from onnx import TensorProto, load_model, numpy_helper
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

from sparseml.onnx.utils import (
    NodeParam,
    SparsityMeasurement,
    calculate_flops,
    check_load_model,
    conv_node_params,
    extract_node_id,
    extract_node_shapes,
    extract_shape,
    gemm_node_params,
    get_init_by_name,
    get_kernel_shape,
    get_node_attributes,
    get_node_by_id,
    get_node_input_nodes,
    get_node_inputs,
    get_node_output_nodes,
    get_node_outputs,
    get_node_params,
    get_nodes_by_input_id,
    get_nodes_by_output_id,
    get_numpy_dtype,
    get_prunable_node_from_foldable,
    get_prunable_nodes,
    is_foldable_node,
    is_prunable_node,
    matmul_node_params,
    model_inputs,
    model_outputs,
    onnx_nodes_sparsities,
)
from sparsezoo import Model, search_models


from tests.sparseml.onnx.helpers import (  # noqa isort: skip
    GENERATE_TEST_FILES,
    onnx_repo_models,
)

RELATIVE_PATH = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture
def simple_onnx_model():
    X = make_tensor_value_info("X", TensorProto.FLOAT, [3, 2])

    Z = make_tensor_value_info("Z", TensorProto.FLOAT, [3, 4])

    node_defs = [
        make_node("Conv", ["X", "node1.weight", "node1.bias"], ["Y"], name="node1"),
        make_node("ReLU", ["Y"], ["Z"], name="node2"),
    ]

    graph_def = make_graph(
        node_defs,
        "test-model",
        [X],
        [Z],
        initializer=[
            numpy_helper.from_array(
                numpy.random.randn(2, 3, 3, 3), name="node1.weight"
            ),
            numpy_helper.from_array(numpy.random.randn(3), name="node1.bias"),
        ],
    )

    model_def = make_model(graph_def)
    return model_def


@pytest.fixture
def foldable_onnx_model():
    X = make_tensor_value_info("X", TensorProto.FLOAT, [3, 2])

    Z = make_tensor_value_info("Z", TensorProto.FLOAT, [3, 4])

    node_defs = [
        make_node("Conv", ["X", "node1.weight", "node1.bias"], ["Y"], name="node1"),
        make_node("batchnormalization", ["Y"], ["Z"], name="node2"),
    ]

    graph_def = make_graph(
        node_defs,
        "test-model",
        [X],
        [Z],
        initializer=[
            numpy_helper.from_array(
                numpy.random.randn(2, 3, 3, 3), name="node1.weight"
            ),
            numpy_helper.from_array(numpy.random.randn(3), name="node1.bias"),
        ],
    )

    model_def = make_model(graph_def)
    return model_def


def get_prunable_onnx_model():
    X = make_tensor_value_info("A", TensorProto.FLOAT, [3, 2])

    Z = make_tensor_value_info("Z", TensorProto.FLOAT, [4])

    node_defs = [
        make_node("Conv", ["A", "node1.weight", "node1.bias"], ["B"], name="node1"),
        make_node("Gemm", ["B", "node2.weight"], ["C"], name="node2"),
        make_node("MatMul", ["C", "node3.weight"], ["D"], name="node3"),
        make_node("ReLU", ["D"], ["Z"], name="node4"),
    ]

    graph_def = make_graph(
        node_defs,
        "test-model",
        [X],
        [Z],
        initializer=[
            numpy_helper.from_array(
                numpy.random.randn(2, 3, 3, 3), name="node1.weight"
            ),
            numpy_helper.from_array(numpy.random.randn(2), name="node1.bias"),
            numpy_helper.from_array(numpy.random.randn(12, 3), name="node2.weight"),
            numpy_helper.from_array(numpy.random.randn(3, 4), name="node3.weight"),
        ],
    )

    model_def = make_model(graph_def)
    return model_def


@pytest.fixture
def prunable_onnx_model():
    return get_prunable_onnx_model()


def test_check_load_model(onnx_repo_models):  # noqa: F811
    model_path = onnx_repo_models.model_path
    loaded_model = load_model(model_path)
    assert loaded_model == check_load_model(model_path)
    assert loaded_model == check_load_model(loaded_model)


@pytest.mark.parametrize(
    "op_type,inputs,outputs", [("Conv", ["X"], ["Y"]), ("Gemm", ["X"], ["Y", "Z"])]
)
def test_extract_node_id(op_type, inputs, outputs):
    node = make_node(op_type, inputs, outputs)
    assert extract_node_id(node) == outputs[0]


def test_get_node_by_id(simple_onnx_model):
    for node in simple_onnx_model.graph.node:
        node_id = extract_node_id(node)
        assert node == get_node_by_id(simple_onnx_model, node_id)

    assert get_node_by_id(simple_onnx_model, "NONE") is None


def test_get_node_by_input_id(simple_onnx_model):
    last_node = simple_onnx_model.graph.node[-1]
    assert get_nodes_by_input_id(simple_onnx_model, "Y") == [last_node]
    assert get_nodes_by_input_id(simple_onnx_model, "NONE") == []


def test_get_node_by_output_id(simple_onnx_model):
    first_node = simple_onnx_model.graph.node[0]
    assert get_nodes_by_output_id(simple_onnx_model, "Y") == [first_node]
    assert get_nodes_by_output_id(simple_onnx_model, "NONE") == []


def test_extract_shape():
    sample_tensor = make_tensor_value_info("X", TensorProto.FLOAT, [3, 2])
    assert extract_shape(sample_tensor) == (3, 2)

    sample_tensor = make_tensor_value_info("X", TensorProto.STRING, None)
    assert extract_shape(sample_tensor) is None


def test_get_numpy_dtype():
    sample_tensor = make_tensor_value_info("X", TensorProto.FLOAT, [3, 2])
    assert get_numpy_dtype(sample_tensor) == numpy.float32

    sample_tensor = make_tensor_value_info("X", TensorProto.INT32, [3, 2])
    assert get_numpy_dtype(sample_tensor) == numpy.int32

    sample_tensor = make_tensor_value_info("X", TensorProto.STRING, None)
    assert get_numpy_dtype(sample_tensor) is None


def test_get_attributes():
    attributes = {
        "kernel": [3, 3],
        "padding": [1, 1, 1, 1],
    }
    node = make_node("Conv", ["X"], ["Y"], **attributes)
    assert get_node_attributes(node) == attributes


def test_get_inputs(simple_onnx_model):
    for node, expected_input in zip(simple_onnx_model.graph.node, [["X"], ["Y"]]):
        assert get_node_inputs(simple_onnx_model, node) == expected_input


def test_get_outputs(simple_onnx_model):
    for node in simple_onnx_model.graph.node:
        assert get_node_outputs(simple_onnx_model, node) == node.output


@pytest.mark.parametrize(
    "node,foldable",
    [
        (make_node("batchnormalization", ["X"], ["Y"]), True),
        (make_node("add", ["X"], ["Y"]), True),
        (make_node("mul", ["X"], ["Y"]), True),
        (make_node("Other", ["X"], ["Y"]), False),
        ("batchnormalization", True),
    ],
)
def test_is_foldable(node, foldable):
    assert is_foldable_node(node) == foldable


def test_get_prunable_node_from_foldable(foldable_onnx_model):
    assert (
        get_prunable_node_from_foldable(
            foldable_onnx_model, foldable_onnx_model.graph.node[-1]
        )
        == foldable_onnx_model.graph.node[0]
    )
    with pytest.raises(ValueError):
        get_prunable_node_from_foldable(
            foldable_onnx_model, foldable_onnx_model.graph.node[0]
        )
    assert (
        get_prunable_node_from_foldable(
            foldable_onnx_model,
            foldable_onnx_model.graph.node[-1],
            traverse_previous=False,
        )
        is None
    )
    assert (
        get_prunable_node_from_foldable(
            foldable_onnx_model, foldable_onnx_model.graph.node[-1], max_node_distance=0
        )
        is None
    )


def test_get_init_by_name(onnx_repo_models):  # noqa: F811
    model = load_model(onnx_repo_models.model_path)
    for init in model.graph.initializer:
        assert init == get_init_by_name(model, init.name)


def test_is_prunable(simple_onnx_model):
    assert is_prunable_node(simple_onnx_model, simple_onnx_model.graph.node[0])
    assert not is_prunable_node(simple_onnx_model, simple_onnx_model.graph.node[-1])


def test_model_inputs(simple_onnx_model):
    assert model_inputs(simple_onnx_model) == list(simple_onnx_model.graph.input)


def test_model_outputs(simple_onnx_model):
    assert model_outputs(simple_onnx_model) == list(simple_onnx_model.graph.output)


def test_get_prunable_nodes(prunable_onnx_model):
    assert (
        get_prunable_nodes(prunable_onnx_model) == prunable_onnx_model.graph.node[:-1]
    )


def test_get_node_input_nodes(simple_onnx_model):
    assert get_node_input_nodes(
        simple_onnx_model, simple_onnx_model.graph.node[-1]
    ) == [simple_onnx_model.graph.node[0]]
    assert (
        get_node_input_nodes(simple_onnx_model, simple_onnx_model.graph.node[0]) == []
    )


def test_get_node_output_nodes(simple_onnx_model):
    assert get_node_output_nodes(
        simple_onnx_model, simple_onnx_model.graph.node[0]
    ) == [simple_onnx_model.graph.node[-1]]
    assert (
        get_node_output_nodes(simple_onnx_model, simple_onnx_model.graph.node[-1]) == []
    )


def test_conv_node_params(prunable_onnx_model):
    conv_node = [
        node for node in prunable_onnx_model.graph.node if node.op_type == "Conv"
    ][0]
    assert conv_node_params(prunable_onnx_model, conv_node, include_values=False) == (
        NodeParam("node1.weight", None),
        NodeParam("node1.bias", None),
    )
    params = conv_node_params(prunable_onnx_model, conv_node)
    assert params[0][1].shape == (2, 3, 3, 3)
    assert params[1][1].shape == (2,)


def test_gemm_node_params(prunable_onnx_model):
    gemm_node = [
        node for node in prunable_onnx_model.graph.node if node.op_type == "Gemm"
    ][0]
    assert gemm_node_params(prunable_onnx_model, gemm_node, include_values=False) == (
        NodeParam("node2.weight", None),
        None,
    )
    params = gemm_node_params(prunable_onnx_model, gemm_node)
    assert params[0][1].shape == (12, 3)


def test_matmul_node_params(prunable_onnx_model):
    matmul_node = [
        node for node in prunable_onnx_model.graph.node if node.op_type == "MatMul"
    ][0]
    assert matmul_node_params(
        prunable_onnx_model, matmul_node, include_values=False
    ) == (NodeParam("node3.weight", None), None)
    params = matmul_node_params(prunable_onnx_model, matmul_node)
    assert params[0][1].shape == (3, 4)


def test_get_node_params(prunable_onnx_model):
    with pytest.raises(ValueError):
        get_node_params(prunable_onnx_model, prunable_onnx_model.graph.node[-1])
    for node, expected_params in zip(
        prunable_onnx_model.graph.node[:-1],
        [
            (NodeParam("node1.weight", None), NodeParam("node1.bias", None)),
            (NodeParam("node2.weight", None), None),
            (NodeParam("node3.weight", None), None),
        ],
    ):
        assert (
            get_node_params(prunable_onnx_model, node, include_values=False)
            == expected_params
        )


def test_onnx_node_sparsities():
    # runs through nearly all other onnx functions imported above as well
    models = search_models(
        domain="cv",
        sub_domain="classification",
        architecture="mobilenet_v1",
        dataset="imagenet",
        framework="pytorch",
        sparse_name="pruned",
        sparse_category="moderate",
        repo="sparseml",
    )
    assert len(models) > 0

    for model in models:
        file_path = model.onnx_model.path

        tot, nodes = onnx_nodes_sparsities(file_path)

        assert len(nodes) == 28

        assert isinstance(tot, SparsityMeasurement)
        assert tot.sparsity > 0.5
        assert tot.params_count == 4209088
        assert tot.params_zero_count > 0.5 * tot.params_count

        for node, val in nodes.items():
            assert isinstance(val, SparsityMeasurement)
            assert val.params_count > 0

            if "sections" not in node and "classifier" not in node:
                continue
            if (
                "depth" in node
                or "sections.0" in node
                or "sections_0" in node
                or "sections.1" in node
                or "sections_1" in node
                or "output" in node
            ):
                continue

            assert val.sparsity > 0.2
            assert val.sparsity < 0.95
            assert val.params_zero_count > 0


@pytest.mark.parametrize(
    "name,stub",
    [
        (
            "resnet50-pq",
            "zoo:cv/classification/resnet_v1-50/"
            "pytorch/sparseml/imagenet/pruned95_quant-none",
        ),
        (
            "mobilenet-p",
            "zoo:cv/classification/mobilenet_v1-1.0"
            "/pytorch/sparseml/imagenet/pruned-moderate",
        ),
    ],
)
def test_extract_node_shape(name, stub):
    model = Model(stub)

    onnx_model = load_model(model.onnx_model.path)
    actual_shapes = extract_node_shapes(onnx_model)

    actual_shapes = {
        k: {"input_shapes": v.input_shapes, "output_shapes": v.output_shapes}
        for k, v in actual_shapes.items()
    }

    data_path = os.path.join(
        RELATIVE_PATH, "test_extract_node_shape_data", name + ".json"
    )
    if GENERATE_TEST_FILES:
        with open(data_path, "w") as fp:
            json.dump(actual_shapes, fp, indent=1)

    with open(data_path) as fp:
        expected_shapes = json.load(fp)

    assert actual_shapes == expected_shapes


@pytest.mark.parametrize(
    "attributes,output",
    [
        ({"kernel": [3, 3]}, [3, 3]),
        ({"kernel_shape": [5, 5]}, [5, 5]),
        ({"stride": [1, 1]}, None),
    ],
)
def test_get_kernel_shape(attributes, output):
    assert get_kernel_shape(attributes) == output


@pytest.mark.parametrize(
    "op_type,input_shape,output_shape,weight_shape,kernel_shape,bias_shape,flops",
    [
        (
            "Add",
            [[1, 3, 15, 15], [1, 3, 15, 15]],
            [[1, 3, 15, 15]],
            None,
            None,
            None,
            3 * 15 * 15,
        ),
        (
            "Mul",
            [[1, 3, 15, 15], [1, 3, 15, 15]],
            [[1, 3, 15, 15]],
            None,
            None,
            None,
            3 * 15 * 15,
        ),
        (
            "Div",
            [[1, 3, 15, 15], [1, 3, 15, 15]],
            [[1, 3, 15, 15]],
            None,
            None,
            None,
            3 * 15 * 15,
        ),
        (
            "Sub",
            [[1, 3, 15, 15], [1, 3, 15, 15]],
            [[1, 3, 15, 15]],
            None,
            None,
            None,
            3 * 15 * 15,
        ),
        (
            "Clip",
            [[1, 3, 15, 15], [1, 3, 15, 15]],
            [[1, 3, 15, 15]],
            None,
            None,
            None,
            3 * 15 * 15,
        ),
        (
            "Relu",
            [[1, 3, 15, 15]],
            [[1, 3, 15, 15]],
            None,
            None,
            None,
            3 * 15 * 15,
        ),
        (
            "LeakyRelu",
            [[1, 3, 15, 15]],
            [[1, 3, 15, 15]],
            None,
            None,
            None,
            3 * 15 * 15,
        ),
        (
            "Sigmoid",
            [[1, 3, 15, 15]],
            [[1, 3, 15, 15]],
            None,
            None,
            None,
            3 * 15 * 15,
        ),
        (
            "Tanh",
            [[1, 3, 15, 15]],
            [[1, 3, 15, 15]],
            None,
            None,
            None,
            3 * 15 * 15,
        ),
        (
            "BatchNormalization",
            [[1, 3, 15, 15]],
            [[1, 3, 15, 15]],
            None,
            None,
            None,
            3 * 15 * 15,
        ),
        (
            "GlobalAveragePool",
            [[1, 3, 15, 15]],
            [[1, 3, 1, 1]],
            None,
            None,
            None,
            3 * 15 * 15,
        ),
        (
            "GlobalMaxPool",
            [[1, 3, 15, 15]],
            [[1, 3, 1, 1]],
            None,
            None,
            None,
            3 * 15 * 15,
        ),
        (
            "MaxPool",
            [[1, 3, 15, 15]],
            [[1, 3, 15, 15]],
            None,
            [3, 3],
            None,
            3 * 3 * 3 * 15 * 15,
        ),
        (
            "AveragePool",
            [[1, 3, 15, 15]],
            [[1, 3, 15, 15]],
            None,
            [3, 3],
            None,
            3 * 3 * 3 * 15 * 15,
        ),
        ("MatMul", [[16]], [[8]], [[16, 8]], None, None, 16 * 8 * 2),
        ("MatMul", [[16]], [[8]], [[16, 8]], None, [[8]], 16 * 8 * 2 + 8),
        (
            "MatMul",
            [[9, 5, 7, 4], [9, 5, 4, 3]],
            [[9, 5, 7, 3]],
            None,
            None,
            None,
            (7 * 3 * (2 * 4 - 1)) * 9 * 5,
        ),
        ("Gemm", [[16]], [[8]], [[16, 8]], None, None, 16 * 8 * 2),
        ("Gemm", [[16]], [[8]], [[16, 8]], None, [[8]], 16 * 8 * 2 + 8),
        (
            "Conv",
            [[16, 4, 3, 3]],
            [[16, 16, 2, 2]],
            [16, 4, 3, 3],
            [3, 3],
            [16],
            3 * 3 * 4 * 16 * 16 * 2 * 2 + 16 * 2 * 2,
        ),
        (
            "Conv",
            [[16, 4, 3, 3]],
            [[16, 16, 2, 2]],
            [16, 4, 3, 3],
            [3, 3],
            None,
            3 * 3 * 4 * 16 * 16 * 2 * 2,
        ),
        (
            "Conv",
            [["batch", 4, 3, 3]],
            [["batch", 16, 2, 2]],
            ["batch", 4, 3, 3],
            [3, 3],
            None,
            3 * 3 * 4 * 16 * 2 * 2,
        ),
    ],
)
def test_calculate_flops(
    op_type, input_shape, output_shape, weight_shape, kernel_shape, bias_shape, flops
):
    assert flops == calculate_flops(
        op_type,
        input_shape=input_shape,
        output_shape=output_shape,
        weight_shape=weight_shape,
        kernel_shape=kernel_shape,
        bias_shape=bias_shape,
    )


@pytest.mark.parametrize(
    "op_type,input_shape,output_shape,weight_shape,kernel_shape,bias_shape",
    [
        (
            "Add",
            [[1, 3, 15, 15], [1, 3, 15, 15]],
            None,
            None,
            None,
            None,
        ),
        (
            "GlobalMaxPool",
            None,
            [[1, 3, 1, 1]],
            None,
            None,
            None,
        ),
        (
            "MaxPool",
            [[1, 3, 15, 15]],
            [[1, 3, 15, 15]],
            None,
            None,
            None,
        ),
        ("Gemm", [[16]], [[8]], None, None, None),
        (
            "MatMul",
            [[9, 5, 7, 4], [9, 5, 5, 3]],
            [[9, 5, 7, 3]],
            None,
            None,
            None,
        ),
    ],
)
def test_calculate_flops_negatives(
    op_type, input_shape, output_shape, weight_shape, kernel_shape, bias_shape
):
    assert (
        calculate_flops(
            op_type,
            input_shape=input_shape,
            output_shape=output_shape,
            weight_shape=weight_shape,
            kernel_shape=kernel_shape,
            bias_shape=bias_shape,
        )
        is None
    )
