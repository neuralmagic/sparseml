import os

import pytest
import torch
from sparseml.pytorch.optim import ModulePruningAnalyzer
from sparseml.pytorch.utils import get_layer
from tests.pytorch.helpers import LinearNet
from torch.nn import Linear, Parameter, ReLU, Sequential


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def _test_const(module, name, param_name):
    layer = get_layer(name, module)
    analyzer = ModulePruningAnalyzer(layer, name, param_name)
    assert analyzer.module == layer
    assert analyzer.name == name
    assert analyzer.param_name == param_name
    assert analyzer.tag == "{}.{}".format(name, param_name)
    assert isinstance(analyzer.param, Parameter)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "module,name,param_name",
    [
        (LinearNet(), LinearNet.layer_descs()[0].name, "weight"),
        (LinearNet(), LinearNet.layer_descs()[0].name, "bias"),
        (LinearNet(), LinearNet.layer_descs()[2].name, "weight"),
        (LinearNet(), LinearNet.layer_descs()[2].name, "bias"),
    ],
)
def test_const(module, name, param_name):
    _test_const(module, name, param_name)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "module,name,param_name",
    [
        (LinearNet(), LinearNet.layer_descs()[0].name, "weight"),
        (LinearNet(), LinearNet.layer_descs()[0].name, "bias"),
        (LinearNet(), LinearNet.layer_descs()[2].name, "weight"),
        (LinearNet(), LinearNet.layer_descs()[2].name, "bias"),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_const_cuda(module, name, param_name):
    module = module.to("cuda")
    _test_const(module, name, param_name)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "module,layers,param_name",
    [
        (
            LinearNet(),
            [LinearNet.layer_descs()[0].name, LinearNet.layer_descs()[2].name],
            "weight",
        ),
        (
            LinearNet(),
            [LinearNet.layer_descs()[0].name, LinearNet.layer_descs()[2].name],
            "bias",
        ),
        (LinearNet(), [LinearNet.layer_descs()[2].name], "weight"),
        (LinearNet(), [LinearNet.layer_descs()[2].name], "bias"),
    ],
)
def test_analyze_layers(module, layers, param_name):
    analyzers = ModulePruningAnalyzer.analyze_layers(module, layers, param_name)
    assert len(analyzers) == len(layers)
    assert isinstance(analyzers[-1], ModulePruningAnalyzer)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "module,layers,param_name",
    [
        (
            LinearNet(),
            [LinearNet.layer_descs()[0].name, LinearNet.layer_descs()[2].name],
            "weight",
        ),
        (
            LinearNet(),
            [LinearNet.layer_descs()[0].name, LinearNet.layer_descs()[2].name],
            "bias",
        ),
        (LinearNet(), [LinearNet.layer_descs()[2].name], "weight"),
        (LinearNet(), [LinearNet.layer_descs()[2].name], "bias"),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_analyze_layers_cuda(module, layers, param_name):
    module = module.to("cuda")
    analyzers = ModulePruningAnalyzer.analyze_layers(module, layers, param_name)
    assert len(analyzers) == len(layers)
    assert isinstance(analyzers[-1], ModulePruningAnalyzer)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "module,name,param_name,param_data,expected_sparsity",
    [
        (
            LinearNet(),
            LinearNet.layer_descs()[0].name,
            "weight",
            torch.zeros(
                LinearNet.layer_descs()[0].input_size[0],
                LinearNet.layer_descs()[0].output_size[0],
            ),
            1.0,
        ),
        (
            LinearNet(),
            LinearNet.layer_descs()[0].name,
            "weight",
            torch.ones(
                LinearNet.layer_descs()[0].input_size[0],
                LinearNet.layer_descs()[0].output_size[0],
            ),
            0.0,
        ),
        (
            LinearNet(),
            LinearNet.layer_descs()[0].name,
            "weight",
            torch.randn(
                LinearNet.layer_descs()[0].input_size[0],
                LinearNet.layer_descs()[0].output_size[0],
            ),
            0.0,
        ),
    ],
)
def test_param_sparsity(module, name, param_name, param_data, expected_sparsity):
    layer = get_layer(name, module)
    analyzer = ModulePruningAnalyzer(layer, name, param_name)
    analyzer.param.data = param_data

    assert torch.sum((analyzer.param_sparsity - expected_sparsity).abs()) < 0.00001
