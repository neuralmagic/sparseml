import pytest

import os
import torch
import torch.nn.functional as TF
from torch.optim import SGD
from torch.utils.data import DataLoader

from neuralmagicML.pytorch.recal import lr_loss_sensitivity
from neuralmagicML.pytorch.utils import LossWrapper

from tests.pytorch.helpers import MLPNet, ConvNet, MLPDataset, ConvDataset


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
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
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
        model, data, loss, "cpu", samples_per_measurement,
    )


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
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
        model, data, loss, "cuda", samples_per_measurement,
    )
