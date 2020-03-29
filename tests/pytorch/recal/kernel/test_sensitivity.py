import pytest

import torch
from torch.nn import Sequential, Linear, ReLU, Conv2d
import torch.nn.functional as TF
from torch.utils.data import Dataset

from neuralmagicML.pytorch.recal import one_shot_ks_loss_sensitivity
from neuralmagicML.pytorch.utils import LossWrapper


TEST_MODEL = Sequential(
    Linear(8, 16), ReLU(), Linear(16, 32), ReLU(), Linear(32, 64), ReLU(), Linear(64, 1)
)


class DatasetImpl(Dataset):
    def __init__(self, length: int):
        self._length = length
        self._x_feats = [torch.randn(8) for _ in range(length)]
        self._y_labs = [torch.randn(1) for _ in range(length)]

    def __getitem__(self, index: int):
        return self._x_feats[index], self._y_labs[index]

    def __len__(self) -> int:
        return self._length


TEST_CNN_MODEL = Sequential(
    Conv2d(8, 16, 3, padding=3, stride=2),
    ReLU(),
    Conv2d(16, 32, 3, padding=3, stride=2),
    ReLU(),
    Conv2d(32, 64, 3, padding=3, stride=2),
    ReLU(),
    Conv2d(64, 1, 3, padding=3, stride=2),
    ReLU(),
)


class CNNDatasetImpl(Dataset):
    def __init__(self, length: int):
        self._length = length
        self._x_feats = [torch.randn(8, 16, 16) for _ in range(length)]
        self._y_labs = [torch.randn(1, 1, 1) for _ in range(length)]

    def __getitem__(self, index: int):
        return self._x_feats[index], self._y_labs[index]

    def __len__(self) -> int:
        return self._length


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
        (TEST_MODEL, DatasetImpl(1000), LossWrapper(TF.mse_loss), 16, 100),
        (TEST_CNN_MODEL, CNNDatasetImpl(1000), LossWrapper(TF.mse_loss), 16, 100),
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
        (TEST_MODEL, DatasetImpl(1000), LossWrapper(TF.mse_loss), 16, 100),
        (TEST_CNN_MODEL, CNNDatasetImpl(1000), LossWrapper(TF.mse_loss), 16, 100),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_module_ks_sensitivity_analysis_one_shot_cuda(
    model, dataset, loss_fn, batch_size, samples_per_measurement
):
    _test_one_shot_ks_loss_sensitivity_helper(
        model, dataset, loss_fn, batch_size, samples_per_measurement, "cuda"
    )
