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
from typing import Tuple

import pytest
import torch
from torch.nn import Linear, Module
from torch.nn.modules.conv import _ConvNd
from torchvision.models import resnet50

from sparseml.pytorch.optim import ModuleAnalyzer
from tests.sparseml.pytorch.helpers import ConvNet, MLPNet


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "model,input_shape,name,params,prunable_params,execution_order,flops,total_flops",
    [
        (
            MLPNet(),
            MLPNet.layer_descs()[0].input_size,
            None,
            2800,
            2688,
            0,
            5600,
            5600,
        ),
        (
            MLPNet(),
            MLPNet.layer_descs()[0].input_size,
            MLPNet.layer_descs()[2].name,
            544,
            512,
            4,
            1056,
            1056,
        ),
        (
            MLPNet(),
            MLPNet.layer_descs()[0].input_size,
            MLPNet.layer_descs()[3].name,
            0,
            0,
            5,
            32,
            32,
        ),
        (
            ConvNet(),
            ConvNet.layer_descs()[0].input_size,
            None,
            5418,
            5360,
            0,
            632564,
            632564,
        ),
        (
            ConvNet(),
            ConvNet.layer_descs()[0].input_size,
            ConvNet.layer_descs()[2].name,
            4640,
            4608,
            4,
            453152,
            453152,
        ),
        (
            resnet50(),
            (3, 224, 224),
            None,
            25557032,
            25529472,
            0,
            8208826344,
            # RN50 has a single ReLU used multiple
            # times, so total is not equal to single-pass
            # FLOPs, even for a single step.
            8212112872,
        ),
    ],
)
def test_analyzer(
    model: Module,
    input_shape: Tuple[int],
    name: str,
    params: int,
    prunable_params: int,
    execution_order: int,
    flops: int,
    total_flops: int,
):
    # Make sure we don't accidentally have 0 weights in a
    # 'dense' model. In real life it's fine, but here it would
    # throw off the expected result.
    def init_weights(m):
        if isinstance(m, Linear) or isinstance(m, _ConvNd):
            m.weight.data.fill_(0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    model.apply(init_weights)
    analyzer = ModuleAnalyzer(model, enabled=True, ignore_zero=True)
    tens = torch.randn(1, *input_shape)
    out = model(tens)
    analyzer.enabled = False
    out = model(tens)
    assert len(out)

    desc = analyzer.layer_desc(name)
    assert desc.params == params
    assert desc.prunable_params == prunable_params
    assert desc.zeroed_params == 0
    assert desc.execution_order == execution_order
    assert desc.flops == flops
    assert desc.total_flops == total_flops


@pytest.mark.parametrize(
    "model,input_shape,name,params,prunable_params,zeroed_params,execution_order,flops,total_flops",  # noqa: E501
    [
        (
            MLPNet(),
            MLPNet.layer_descs()[0].input_size,
            None,
            2800,
            2688,
            56,
            0,
            5488,
            5488,
        ),
        (
            MLPNet(),
            MLPNet.layer_descs()[0].input_size,
            MLPNet.layer_descs()[2].name,
            544,
            512,
            16,
            4,
            1024,
            1024,
        ),
        (
            MLPNet(),
            MLPNet.layer_descs()[0].input_size,
            MLPNet.layer_descs()[3].name,
            0,
            0,
            0,
            5,
            32,
            32,
        ),
        (
            ConvNet(),
            ConvNet.layer_descs()[0].input_size,
            None,
            5418,
            5360,
            203,
            0,
            607804,
            607804,
        ),
        (
            ConvNet(),
            ConvNet.layer_descs()[0].input_size,
            ConvNet.layer_descs()[2].name,
            4640,
            4608,
            144,
            4,
            439040,
            439040,
        ),
        (
            resnet50(),
            (3, 224, 224),
            None,
            25557032,
            25529472,
            54931,
            0,
            8165194368,
            # RN50 has a single ReLU used multiple
            # times, so total is not equal to single-pass
            # FLOPs, even for a single step.
            8168480896,
        ),
    ],
)
def test_analyzer_sparse(
    model: Module,
    input_shape: Tuple[int],
    name: str,
    params: int,
    prunable_params: int,
    zeroed_params: int,
    execution_order: int,
    flops: int,
    total_flops: int,
):
    def init_weights(m):
        if isinstance(m, Linear) or isinstance(m, _ConvNd):
            m.weight.data.fill_(0.01)
            # Set some weights to 0
            m.weight.data[0] = 0
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    model.apply(init_weights)
    analyzer = ModuleAnalyzer(model, enabled=True, ignore_zero=True)
    tens = torch.randn(1, *input_shape)
    out = model(tens)
    analyzer.enabled = False
    out = model(tens)
    assert len(out)

    desc = analyzer.layer_desc(name)
    assert desc.params == params
    assert desc.prunable_params == prunable_params
    assert desc.zeroed_params == zeroed_params
    assert desc.execution_order == execution_order
    assert desc.flops == flops
    assert desc.total_flops == total_flops
