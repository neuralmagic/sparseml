import pytest

import sys
import math
import torch
from torch.nn import Sequential, Linear, ReLU, Conv2d
import torch.nn.functional as TF
from torch.utils.data import Dataset

from neuralmagicML.pytorch.recal import (
    ModuleParamKSSensitivity,
    KSSensitivityProgress,
    ModuleKSSensitivityAnalysis,
)
from neuralmagicML.pytorch.utils import LossWrapper


@pytest.mark.parametrize(
    "name,param_name,type_,execution_order,measured,expected_integral",
    [
        (
            "layer.name",
            "weight",
            "linear",
            0,
            [(0.05, 0.0), (0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (0.95, 0.0)],
            0.0,
        ),
        (
            "layer.name",
            "weight",
            "conv",
            0,
            [(0.05, 1.0), (0.25, 1.0), (0.5, 1.0), (0.75, 1.0), (0.95, 1.0)],
            1.0,
        ),
        (
            "layer.name",
            "weight",
            "conv",
            0,
            [(0.05, 1.0), (0.25, 1.5), (0.5, 2.0), (0.75, 3.75), (0.95, 6.0)],
            2.7312498092651367,
        ),
    ],
)
def test_module_param_ks_sensitivity(
    name, param_name, type_, execution_order, measured, expected_integral
):
    sens = ModuleParamKSSensitivity(name, param_name, type_, execution_order, measured)
    assert sens.name == name
    assert sens.param_name == param_name
    assert sens.type_ == type_
    assert sens.execution_order == execution_order
    assert abs(sens.integral - expected_integral) < sys.float_info.epsilon


@pytest.mark.parametrize(
    "layer_index,layer_name,layers,sparsity_index,sparsity_levels,measurement_step,samples_per_measurement",
    [
        (
            0,
            "layer.name",
            ["layer.name", "layer2.name"],
            0,
            [0.2, 0.5, 0.7, 0.9, 0.95],
            0,
            100,
        )
    ],
)
def test_ks_sensitivity_progress(
    layer_index,
    layer_name,
    layers,
    sparsity_index,
    sparsity_levels,
    measurement_step,
    samples_per_measurement,
):
    progress = KSSensitivityProgress(
        layer_index,
        layer_name,
        layers,
        sparsity_index,
        sparsity_levels,
        measurement_step,
        samples_per_measurement,
    )
    assert progress.layer_index == layer_index
    assert progress.layer_name == layer_name
    assert progress.layers == layers
    assert progress.sparsity_index == sparsity_index
    assert progress.sparsity_levels == sparsity_levels
    assert progress.measurement_step == measurement_step
    assert progress.samples_per_measurement == samples_per_measurement


TEST_MODEL = Sequential(
    Linear(8, 16), ReLU(), Linear(16, 32), ReLU(), Linear(32, 64), ReLU(), Linear(64, 1)
)


class TestingDataset(Dataset):
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


class TestingCNNDataset(Dataset):
    def __init__(self, length: int):
        self._length = length
        self._x_feats = [torch.randn(8, 16, 16) for _ in range(length)]
        self._y_labs = [torch.randn(1, 1, 1) for _ in range(length)]

    def __getitem__(self, index: int):
        return self._x_feats[index], self._y_labs[index]

    def __len__(self) -> int:
        return self._length


def _test_module_ks_sensitivity_analysis_one_shot(
    model, dataset, loss_fn, batch_size, samples_per_measurement, device
):
    analysis = ModuleKSSensitivityAnalysis(model, dataset, loss_fn)
    progress_counter = 0

    def _progress_hook(_progress: KSSensitivityProgress):
        nonlocal progress_counter
        progress_counter += 1

    handle = analysis.register_progress_hook(_progress_hook)
    results = analysis.run_one_shot(device, batch_size, samples_per_measurement)
    assert len(results) > 0
    expected = (
        (1 + math.ceil(samples_per_measurement / batch_size))
        * len(results[-1].measured)
        + 1
    ) * len(results)
    assert expected == progress_counter
    handle.remove()

    for res in results:
        assert res.integral > 0.0


@pytest.mark.parametrize(
    "model,dataset,loss_fn,batch_size,samples_per_measurement",
    [
        (TEST_MODEL, TestingDataset(1000), LossWrapper(TF.mse_loss), 16, 100),
        (TEST_CNN_MODEL, TestingCNNDataset(1000), LossWrapper(TF.mse_loss), 16, 100),
    ],
)
def test_module_ks_sensitivity_analysis_one_shot(
    model, dataset, loss_fn, batch_size, samples_per_measurement
):
    _test_module_ks_sensitivity_analysis_one_shot(
        model, dataset, loss_fn, batch_size, samples_per_measurement, "cpu"
    )


@pytest.mark.parametrize(
    "model,dataset,loss_fn,batch_size,samples_per_measurement",
    [
        (TEST_MODEL, TestingDataset(1000), LossWrapper(TF.mse_loss), 16, 100),
        (TEST_CNN_MODEL, TestingCNNDataset(1000), LossWrapper(TF.mse_loss), 16, 100),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_module_ks_sensitivity_analysis_one_shot_cuda(
    model, dataset, loss_fn, batch_size, samples_per_measurement
):
    _test_module_ks_sensitivity_analysis_one_shot(
        model, dataset, loss_fn, batch_size, samples_per_measurement, "cuda"
    )
