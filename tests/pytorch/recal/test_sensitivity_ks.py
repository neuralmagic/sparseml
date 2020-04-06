import pytest

import torch
import torch.nn.functional as TF

from neuralmagicML.pytorch.recal import one_shot_ks_loss_sensitivity
from neuralmagicML.pytorch.utils import LossWrapper

from tests.pytorch.helpers import MLPNet, ConvNet, MLPDataset, ConvDataset


def _test_one_shot_ks_loss_sensitivity_helper(
    model, dataset, loss_fn, batch_size, samples_per_measurement, device
):
    analysis = one_shot_ks_loss_sensitivity(
        model, dataset, loss_fn, device, batch_size, samples_per_measurement
    )

    for res in analysis.results:
        assert res.integral > 0.0


@pytest.mark.parametrize(
    "model,dataset,loss_fn,batch_size,samples_per_measurement",
    [
        (MLPNet(), MLPDataset(), LossWrapper(TF.mse_loss), 16, 100),
        (ConvNet(), ConvDataset(), LossWrapper(TF.mse_loss), 16, 100),
    ],
)
def test_module_ks_sensitivity_analysis_one_shot(
    model, dataset, loss_fn, batch_size, samples_per_measurement
):
    _test_one_shot_ks_loss_sensitivity_helper(
        model, dataset, loss_fn, batch_size, samples_per_measurement, "cpu"
    )


@pytest.mark.parametrize(
    "model,dataset,loss_fn,batch_size,samples_per_measurement",
    [
        (MLPNet(), MLPDataset(), LossWrapper(TF.mse_loss), 16, 100),
        (ConvNet(), ConvDataset(), LossWrapper(TF.mse_loss), 16, 100),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_module_ks_sensitivity_analysis_one_shot_cuda(
    model, dataset, loss_fn, batch_size, samples_per_measurement
):
    _test_one_shot_ks_loss_sensitivity_helper(
        model, dataset, loss_fn, batch_size, samples_per_measurement, "cuda"
    )
