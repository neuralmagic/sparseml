import pytest

import torch
from torch.nn import Parameter, Sequential, Linear, ReLU

from neuralmagicML.recal import ModuleKSAnalyzer
from neuralmagicML.utils import get_layer


TEST_MODULE = Sequential(
    Linear(8, 16),
    ReLU(),
    Linear(16, 32),
    ReLU(),
    Sequential(Linear(32, 64), ReLU(), Linear(64, 64), ReLU()),
    Linear(64, 1),
    ReLU(),
)


def _test_const(module, name, param_name):
    layer = get_layer(name, module)
    analyzer = ModuleKSAnalyzer(layer, name, param_name)
    assert analyzer.module == layer
    assert analyzer.name == name
    assert analyzer.param_name == param_name
    assert analyzer.tag == "{}.{}".format(name, param_name)
    assert isinstance(analyzer.param, Parameter)


@pytest.mark.parametrize(
    "module,name,param_name",
    [
        (TEST_MODULE, "0", "weight"),
        (TEST_MODULE, "0", "bias"),
        (TEST_MODULE, "4.0", "weight"),
        (TEST_MODULE, "4.2", "bias"),
    ],
)
@pytest.mark.device_cpu
def test_const(module, name, param_name):
    _test_const(module, name, param_name)


@pytest.mark.parametrize(
    "module,name,param_name",
    [
        (TEST_MODULE, "0", "weight"),
        (TEST_MODULE, "0", "bias"),
        (TEST_MODULE, "4.0", "weight"),
        (TEST_MODULE, "4.2", "bias"),
    ],
)
@pytest.mark.device_cuda
def test_const_cuda(module, name, param_name):
    module = module.to("cuda")
    _test_const(module, name, param_name)


@pytest.mark.parametrize(
    "module,layers,param_name",
    [
        (TEST_MODULE, ["0", "2", "4.2"], "weight"),
        (TEST_MODULE, ["0"], "bias"),
        (TEST_MODULE, ["4.2"], "weight"),
        (TEST_MODULE, ["0", "2", "4.2"], "bias"),
    ],
)
@pytest.mark.device_cpu
def test_analyze_layers(module, layers, param_name):
    analyzers = ModuleKSAnalyzer.analyze_layers(module, layers, param_name)
    assert len(analyzers) == len(layers)
    assert isinstance(analyzers[-1], ModuleKSAnalyzer)


@pytest.mark.parametrize(
    "module,layers,param_name",
    [
        (TEST_MODULE, ["0", "2", "4.2"], "weight"),
        (TEST_MODULE, ["0"], "bias"),
        (TEST_MODULE, ["4.2"], "weight"),
        (TEST_MODULE, ["0", "2", "4.2"], "bias"),
    ],
)
@pytest.mark.device_cuda
def test_analyze_layers_cuda(module, layers, param_name):
    module = module.to("cuda")
    analyzers = ModuleKSAnalyzer.analyze_layers(module, layers, param_name)
    assert len(analyzers) == len(layers)
    assert isinstance(analyzers[-1], ModuleKSAnalyzer)


@pytest.mark.parametrize(
    "module,name,param_name,param_data,expected_sparsity",
    [
        (TEST_MODULE, "0", "weight", torch.zeros(16, 8), 1.0),
        (TEST_MODULE, "0", "weight", torch.ones(16, 8), 0.0),
        (TEST_MODULE, "0", "weight", torch.randn(16, 8), 0.0),
        (
            TEST_MODULE,
            "0",
            "bias",
            torch.tensor(
                [
                    0.0,
                    1.0,
                    3.0,
                    0.0,
                    5.0,
                    3.0,
                    1.0,
                    0.0,
                    7.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    4.0,
                    0.0,
                    0.0,
                ]
            ),
            0.4375,
        ),
    ],
)
@pytest.mark.device_cpu
def test_param_sparsity(module, name, param_name, param_data, expected_sparsity):
    layer = get_layer(name, module)
    analyzer = ModuleKSAnalyzer(layer, name, param_name)
    analyzer.param.data = param_data

    assert torch.sum((analyzer.param_sparsity - expected_sparsity).abs()) < 0.00001
