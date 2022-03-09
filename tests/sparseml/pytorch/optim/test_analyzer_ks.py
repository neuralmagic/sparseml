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

import os

import pytest
import torch
from torch.nn import Parameter

from sparseml.pytorch.optim import ModulePruningAnalyzer
from sparseml.pytorch.utils import get_layer
from tests.sparseml.pytorch.helpers import LinearNet


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
