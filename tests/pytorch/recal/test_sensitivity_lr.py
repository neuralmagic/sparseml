import pytest

import os
import torch
import torch.nn.functional as TF

from neuralmagicML.pytorch.recal import (
    lr_loss_sensitivity,
    default_exponential_check_lrs,
)
from neuralmagicML.pytorch.utils import LossWrapper

from tests.pytorch.helpers import MLPNet, ConvNet, MLPDataset, ConvDataset


def _test_lr_sensitivity(
    model, dataset, loss_fn, batch_size, batches_per_measurement, device
):
    lrs = default_exponential_check_lrs()
    analysis = lr_loss_sensitivity(
        model, dataset, loss_fn, device, batch_size, batches_per_measurement, lrs,
    )

    for lr, val in analysis:
        assert lr in lrs
        assert val > 0


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "model,dataset,loss_fn,batch_size,samples_per_measurement",
    [(MLPNet(), MLPDataset(), LossWrapper(TF.mse_loss), 16, 10)],
)
def test_module_ks_sensitivity_analysis_one_shot(
    model, dataset, loss_fn, batch_size, samples_per_measurement
):
    _test_lr_sensitivity(
        model, dataset, loss_fn, batch_size, samples_per_measurement, "cpu"
    )


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "model,dataset,loss_fn,batch_size,samples_per_measurement",
    [
        (MLPNet(), MLPDataset(), LossWrapper(TF.mse_loss), 16, 10),
        (ConvNet(), ConvDataset(), LossWrapper(TF.mse_loss), 16, 10),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_module_ks_sensitivity_analysis_one_shot_cuda(
    model, dataset, loss_fn, batch_size, samples_per_measurement
):
    _test_lr_sensitivity(
        model, dataset, loss_fn, batch_size, samples_per_measurement, "cuda"
    )
