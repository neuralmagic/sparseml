import pytest

import os
import torch
import torch.nn.functional as TF
from torch.utils.data import DataLoader

from neuralmagicML.pytorch.recal import (
    approx_ks_loss_sensitivity,
    one_shot_ks_loss_sensitivity,
)
from neuralmagicML.pytorch.utils import LossWrapper

from tests.pytorch.helpers import MLPNet, ConvNet, MLPDataset, ConvDataset


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "model", [MLPNet(), ConvNet()],
)
def test_module_ks_sensitivity_analysis_one_shot(model):
    analysis = approx_ks_loss_sensitivity(model)

    for res in analysis.results:
        assert res.name
        assert isinstance(res.index, int)
        assert len(res.sparse_measurements) > 0
        assert len(res.averages) > 0
        assert res.sparse_average > 0
        assert res.sparse_integral > 0
        assert res.sparse_comparison > 0


def _test_one_shot_ks_loss_sensitivity_helper(
    model, data, loss, device, steps_per_measurement
):
    analysis = one_shot_ks_loss_sensitivity(
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
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "model,dataset,loss,batch_size,steps_per_measurement",
    [
        (MLPNet(), MLPDataset(), LossWrapper(TF.mse_loss), 16, 100),
        (ConvNet(), ConvDataset(), LossWrapper(TF.mse_loss), 16, 100),
    ],
)
def test_module_ks_sensitivity_analysis_one_shot(
    model, dataset, loss, batch_size, steps_per_measurement
):
    data = DataLoader(dataset, batch_size)
    _test_one_shot_ks_loss_sensitivity_helper(
        model, data, loss, "cpu", steps_per_measurement,
    )


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
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
        model, data, loss, "cuda", steps_per_measurement,
    )
