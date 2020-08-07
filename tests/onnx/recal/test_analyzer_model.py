import pytest

from onnx import load_model

from neuralmagicML.onnx.recal import ModelAnalyzer, NodeAnalyzer

from tests.onnx.helpers import analyzer_models


def test_node_analyzer_kwargs():
    kwargs = {
        "id": "id",
        "op_type": "Conv",
        "input_names": ["in1", "in2"],
        "output_names": ["out1"],
        "input_shapes": [[1, 16, 16], [1, 32, 32]],
        "output_shapes": [[1, 18, 18]],
        "params": 100,
        "prunable": True,
        "prunable_params_zeroed": 40,
        "weight_name": "conv.section.1.weight",
        "weight_shape": [16, 16, 16],
        "bias_name": "conv.section.1.bias",
        "bias_shape": [16],
        "attributes": {"kernel": [3, 3]},
    }

    node = NodeAnalyzer(model=None, node=None, **kwargs)
    for key in kwargs:
        if key == "id":
            assert node.id_ == kwargs[key]
        else:
            assert getattr(node, key) == kwargs[key]

    assert node.prunable
    assert node.prunable_params == 16 * 16 * 16

    # TODO flops tests when implemented


def test_mode_analyzer_json():
    params = {
        "nodes": [
            {
                "id": "id",
                "op_type": "Conv",
                "input_names": ["in1", "in2"],
                "output_names": ["out1"],
                "input_shapes": [[1, 16, 16], [1, 32, 32]],
                "output_shapes": [[1, 18, 18]],
                "params": 100,
                "prunable": True,
                "prunable_params_zeroed": 40,
                "weight_name": "conv.section.1.weight",
                "weight_shape": [16, 16, 16],
                "bias_name": "conv.section.1.bias",
                "bias_shape": [16],
                "attributes": {"kernel": 1},
            },
            {
                "id": "id2",
                "op_type": "Gemm",
                "input_names": ["in1", "in2"],
                "output_names": ["out1"],
                "input_shapes": [[16]],
                "output_shapes": [[18]],
                "params": 100,
                "prunable": True,
                "prunable_params_zeroed": 40,
                "weight_name": "conv.section.1.weight",
                "weight_shape": [16, 16, 16],
                "bias_name": "conv.section.1.bias",
                "bias_shape": [16],
                "attributes": {"kernel": 1},
            },
        ]
    }
    nodes = [
        NodeAnalyzer(model=None, node=None, **params["nodes"][0]),
        NodeAnalyzer(model=None, node=None, **params["nodes"][1]),
    ]

    analyzer = ModelAnalyzer.from_dict(params)
    assert sorted(analyzer.nodes, key=lambda node: node.id_) == sorted(
        nodes, key=lambda node: node.id_
    )


def test_model_analyzer(analyzer_models):
    model_path, expected_output = analyzer_models
    analyzer = ModelAnalyzer(model_path)

    analyzer_from_json = ModelAnalyzer.from_dict(expected_output)
    assert analyzer.dict() == expected_output
    assert analyzer == analyzer_from_json
