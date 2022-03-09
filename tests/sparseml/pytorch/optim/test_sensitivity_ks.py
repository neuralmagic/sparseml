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
import torch.nn.functional as TF
from torch.utils.data import DataLoader

from sparseml.pytorch.optim import (
    pruning_loss_sens_magnitude,
    pruning_loss_sens_one_shot,
)
from sparseml.pytorch.utils import LossWrapper
from tests.sparseml.pytorch.helpers import ConvDataset, ConvNet, MLPDataset, MLPNet


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "model",
    [MLPNet(), ConvNet()],
)
def test_module_ks_sensitivity_analysis_one_shot(model):
    analysis = pruning_loss_sens_magnitude(model)

    for res in analysis.results:
        assert res.name
        assert isinstance(res.index, int)
        assert len(res.sparse_measurements) > 0
        assert len(res.averages) > 0
        assert res.sparse_average > 0
        assert res.sparse_integral > 0
        assert res.sparse_comparison() > 0


def _test_one_shot_ks_loss_sensitivity_helper(
    model, data, loss, device, steps_per_measurement
):
    analysis = pruning_loss_sens_one_shot(
        model, data, loss, device, steps_per_measurement
    )

    for res in analysis.results:
        assert res.name
        assert isinstance(res.index, int)
        assert len(res.sparse_measurements) > 0
        assert len(res.averages) > 0
        assert res.sparse_average > 0
        assert res.sparse_integral > 0


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "model,dataset,loss,batch_size,steps_per_measurement",
    [
        (MLPNet(), MLPDataset(), LossWrapper(TF.mse_loss), 16, 100),
        (ConvNet(), ConvDataset(), LossWrapper(TF.mse_loss), 16, 100),
    ],
)
def test_model_ks_sensitivity_analysis_one_shot(
    model, dataset, loss, batch_size, steps_per_measurement
):
    data = DataLoader(dataset, batch_size)
    _test_one_shot_ks_loss_sensitivity_helper(
        model,
        data,
        loss,
        "cpu",
        steps_per_measurement,
    )


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "model,dataset,loss,batch_size,steps_per_measurement",
    [
        (MLPNet(), MLPDataset(), LossWrapper(TF.mse_loss), 16, 100),
        (ConvNet(), ConvDataset(), LossWrapper(TF.mse_loss), 16, 100),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_module_ks_sensitivity_analysis_one_shot_cuda(
    model, dataset, loss, batch_size, steps_per_measurement
):
    data = DataLoader(dataset, batch_size)
    model = model.to("cuda")
    _test_one_shot_ks_loss_sensitivity_helper(
        model,
        data,
        loss,
        "cuda",
        steps_per_measurement,
    )
