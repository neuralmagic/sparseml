import pytest

from onnx import load_model
from neuralmagicML.onnx.utils import (
    extract_node_id,
    extract_node_shapes,
    get_node_by_id,
    get_init_by_name,
    NodeParam,
    conv_node_params,
    gemm_node_params,
    matmul_node_params,
    get_node_params,
    get_prunable_nodes,
    onnx_nodes_sparsities,
    SparsityMeasurement,
    get_kernel_shape,
    calculate_flops,
)
from neuralmagicML.utils import available_models

from tests.onnx.helpers import extract_node_models


def test_onnx_node_sparsities():
    # runs through nearly all other onnx functions imported above as well
    models = available_models(
        domains=["cv"],
        sub_domains=["classification"],
        architectures=["mobilenet-v1"],
        datasets=["imagenet"],
        descs=["recal-perf"],
    )
    assert len(models) > 0

    for model in models:
        file_path = model.download_onnx_file()

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
            ):
                continue

            assert val.sparsity > 0.2
            assert val.sparsity < 0.95
            assert val.params_zero_count > 0


def test_extract_node_shape(extract_node_models):
    model_path, expected_output = extract_node_models
    onnx_model = load_model(model_path)
    node_shapes = extract_node_shapes(onnx_model)
    for node in node_shapes:
        assert node_shapes[node].input_shapes == expected_output[node][0]
        assert node_shapes[node].output_shapes == expected_output[node][1]


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
        ("Relu", [[1, 3, 15, 15]], [[1, 3, 15, 15]], None, None, None, 3 * 15 * 15,),
        (
            "LeakyRelu",
            [[1, 3, 15, 15]],
            [[1, 3, 15, 15]],
            None,
            None,
            None,
            3 * 15 * 15,
        ),
        ("Sigmoid", [[1, 3, 15, 15]], [[1, 3, 15, 15]], None, None, None, 3 * 15 * 15,),
        ("Tanh", [[1, 3, 15, 15]], [[1, 3, 15, 15]], None, None, None, 3 * 15 * 15,),
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
        ("Add", [[1, 3, 15, 15], [1, 3, 15, 15]], None, None, None, None,),
        ("GlobalMaxPool", None, [[1, 3, 1, 1]], None, None, None,),
        ("MaxPool", [[1, 3, 15, 15]], [[1, 3, 15, 15]], None, None, None,),
        ("Gemm", [[16]], [[8]], None, None, None),
        ("MatMul", [[9, 5, 7, 4], [9, 5, 5, 3]], [[9, 5, 7, 3]], None, None, None,),
    ],
)
def test_calculate_flops_negatives(
    op_type, input_shape, output_shape, weight_shape, kernel_shape, bias_shape
):
    assert calculate_flops(
        op_type,
        input_shape=input_shape,
        output_shape=output_shape,
        weight_shape=weight_shape,
        kernel_shape=kernel_shape,
        bias_shape=bias_shape,
    ) is None
