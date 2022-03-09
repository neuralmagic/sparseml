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
from torch.optim import SGD
from torch.utils.data import DataLoader

from sparseml.pytorch.optim import lr_loss_sensitivity
from sparseml.pytorch.utils import LossWrapper
from tests.sparseml.pytorch.helpers import ConvDataset, ConvNet, MLPDataset, MLPNet


def _test_lr_sensitivity(model, data, loss, device, steps_per_measurement):
    analysis = lr_loss_sensitivity(
        model,
        data,
        loss,
        SGD(model.parameters(), lr=1.0),
        device,
        steps_per_measurement,
    )

    for res in analysis.results:
        assert "lr" in res
        assert "loss_measurements" in res
        assert "loss_avg" in res


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "model,dataset,loss,batch_size,samples_per_measurement",
    [(MLPNet(), MLPDataset(), LossWrapper(TF.mse_loss), 16, 10)],
)
def test_module_ks_sensitivity_analysis_one_shot(
    model, dataset, loss, batch_size, samples_per_measurement
):
    data = DataLoader(dataset, batch_size)
    _test_lr_sensitivity(
        model,
        data,
        loss,
        "cpu",
        samples_per_measurement,
    )


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "model,dataset,loss,batch_size,samples_per_measurement",
    [
        (MLPNet(), MLPDataset(), LossWrapper(TF.mse_loss), 16, 10),
        (ConvNet(), ConvDataset(), LossWrapper(TF.mse_loss), 16, 10),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_module_ks_sensitivity_analysis_one_shot_cuda(
    model, dataset, loss, batch_size, samples_per_measurement
):
    data = DataLoader(dataset, batch_size)
    model = model.to("cuda")
    _test_lr_sensitivity(
        model,
        data,
        loss,
        "cuda",
        samples_per_measurement,
    )
