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
from torch import Tensor

from sparseml.pytorch.optim import ModuleASAnalyzer
from tests.sparseml.pytorch.helpers import MLPNet


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_as_analyzer():
    model = MLPNet()
    layer_descs = [MLPNet.layer_descs()[1], MLPNet.layer_descs()[3]]
    analyzers = ModuleASAnalyzer.analyze_layers(
        model,
        [desc.name for desc in layer_descs],
        dim=None,
        track_inputs_sparsity=True,
        track_outputs_sparsity=True,
        inputs_sample_size=100,
        outputs_sample_size=100,
        enabled=True,
    )

    for _ in range(10):
        model(torch.randn(1, MLPNet.layer_descs()[0].input_size[0]))

    for analyzer in analyzers:
        assert isinstance(analyzer.inputs_sparsity_mean, Tensor)
        assert analyzer.inputs_sparsity_mean == 0
        assert isinstance(analyzer.inputs_sparsity_std, Tensor)
        assert analyzer.inputs_sparsity_std == 0

        assert isinstance(analyzer.inputs_sample_mean, Tensor)
        assert isinstance(analyzer.inputs_sample_std, Tensor)
        assert analyzer.inputs_sample_std > 0

        assert isinstance(analyzer.outputs_sparsity_mean, Tensor)
        assert analyzer.outputs_sparsity_mean > 0
        assert isinstance(analyzer.outputs_sparsity_std, Tensor)
        assert analyzer.outputs_sparsity_std > 0

        assert isinstance(analyzer.outputs_sample_mean, Tensor)
        assert analyzer.outputs_sample_mean > 0
        assert isinstance(analyzer.outputs_sample_std, Tensor)
        assert analyzer.outputs_sample_std > 0
