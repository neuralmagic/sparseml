import pytest
import os

import sys
import torch
from torch.nn import Linear, Conv2d

from neuralmagicML.pytorch.recal import ModuleParamKSMask
from neuralmagicML.pytorch.utils import tensor_sparsity


def _test_constructor(layer, param_name):
    mask = ModuleParamKSMask(layer, param_name)
    assert mask.layer == layer
    assert mask.param_name == param_name
    assert not mask.store_init
    assert not mask.store_unmasked
    assert mask.track_grad_mom == -1.0
    assert not mask.enabled


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "layer,param_name",
    [
        (Linear(in_features=8, out_features=64), "weight"),
        (Linear(in_features=8, out_features=64), "bias"),
        (Conv2d(in_channels=3, out_channels=64, kernel_size=3), "weight"),
        (Conv2d(in_channels=3, out_channels=64, kernel_size=3), "bias"),
    ],
)
def test_constructor(layer, param_name):
    _test_constructor(layer, param_name)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "layer,param_name",
    [
        (Linear(in_features=8, out_features=64), "weight"),
        (Conv2d(in_channels=3, out_channels=64, kernel_size=3), "weight"),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_constructor_cuda(layer, param_name):
    layer = layer.to("cuda")
    _test_constructor(layer, param_name)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
def _test_set_param_data(layer, param_name, data):
    mask = ModuleParamKSMask(layer, param_name)
    mask.set_param_data(data)
    assert torch.sum((mask.param_data - data).abs()) < sys.float_info.epsilon


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "layer,param_name,data",
    [
        (Linear(in_features=8, out_features=64), "weight", torch.randn(64, 8)),
        (Linear(in_features=8, out_features=64), "bias", torch.randn(64)),
        (
            Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            "weight",
            torch.randn(64, 3, 3, 3),
        ),
        (
            Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            "bias",
            torch.randn(64),
        ),
    ],
)
def test_set_param_data(layer, param_name, data):
    _test_set_param_data(layer, param_name, data)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "layer,param_name,data",
    [
        (Linear(in_features=8, out_features=64), "weight", torch.randn(64, 8)),
        (
            Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            "weight",
            torch.randn(64, 3, 3, 3),
        ),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_set_param_data_cuda(layer, param_name, data):
    layer = layer.to("cuda")
    data = data.to("cuda")
    _test_set_param_data(layer, param_name, data)


def random_mask(*size, threshold):
    mask = torch.randn(*size)

    return (mask <= threshold).type(torch.float32)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
def _test_set_param_mask(layer, param_name, param_mask):
    mask = ModuleParamKSMask(layer, param_name)
    result = mask.set_param_mask(param_mask)
    res_unmasked = (result == 1.0).type(torch.float32)
    res_masked = (result == -1.0).type(torch.float32)
    res_no_change = (result == 0.0).type(torch.float32)
    mask_ones = (param_mask == 1.0).type(torch.float32)
    mask_zeros = (param_mask == 0.0).type(torch.float32)

    assert torch.sum(res_unmasked.abs()) < sys.float_info.epsilon
    assert torch.sum((res_masked - mask_zeros).abs()) < sys.float_info.epsilon
    assert torch.sum((res_no_change - mask_ones).abs()) < sys.float_info.epsilon

    mask.enabled = True
    mask.apply()
    param_data_zeros = (mask.param_data == 0.0).type("float32")

    assert torch.sum((param_data_zeros - mask_zeros).abs()) < sys.float_info.epsilon


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "layer,param_name,param_mask",
    [
        (Linear(in_features=8, out_features=64), "weight", torch.zeros(64, 8)),
        (Linear(in_features=8, out_features=64), "weight", torch.ones(64, 8)),
        (
            Linear(in_features=8, out_features=64),
            "weight",
            random_mask(64, 8, threshold=0.5),
        ),
        (
            Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            "weight",
            torch.zeros(64, 3, 3, 3),
        ),
        (
            Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            "weight",
            torch.ones(64, 3, 3, 3),
        ),
        (
            Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            "weight",
            random_mask(64, 3, 3, 3, threshold=0.5),
        ),
    ],
)
def test_set_param_mask(layer, param_name, param_mask):
    _test_set_param_data(layer, param_name, param_mask)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "layer,param_name,param_mask",
    [
        (Linear(in_features=8, out_features=64), "weight", torch.zeros(64, 8)),
        (Linear(in_features=8, out_features=64), "weight", torch.ones(64, 8)),
        (
            Linear(in_features=8, out_features=64),
            "weight",
            random_mask(64, 8, threshold=0.5),
        ),
        (
            Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            "weight",
            torch.zeros(64, 3, 3, 3),
        ),
        (
            Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            "weight",
            torch.ones(64, 3, 3, 3),
        ),
        (
            Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            "weight",
            random_mask(64, 3, 3, 3, threshold=0.5),
        ),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_set_param_mask_cuda(layer, param_name, param_mask):
    layer = layer.to("cuda")
    param_mask = param_mask.to("cuda")
    _test_set_param_data(layer, param_name, param_mask)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
def _test_set_param_mask_from_abs_threshold(
    layer, param_name, param, threshold, expected_sparsity
):
    mask = ModuleParamKSMask(layer, param_name)
    mask.set_param_data(param)
    mask.set_param_mask_from_abs_threshold(threshold)
    sparsity = tensor_sparsity(mask.param_mask)
    assert (sparsity - expected_sparsity).abs() < 0.01


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "layer,param_name,param,threshold,expected_sparsity",
    [
        (
            Linear(in_features=256, out_features=512),
            "weight",
            torch.randn(512, 256),
            0.0,
            0.0,
        ),
        (
            Linear(in_features=256, out_features=512),
            "weight",
            torch.randn(512, 256),
            1.0,
            0.682,
        ),
        (
            Linear(in_features=256, out_features=512),
            "weight",
            torch.randn(512, 256),
            2.0,
            0.9545,
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            0.0,
            0.0,
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            1.0,
            0.682,
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            2.0,
            0.9545,
        ),
    ],
)
def test_set_param_mask_from_abs_threshold(
    layer, param_name, param, threshold, expected_sparsity
):
    _test_set_param_mask_from_abs_threshold(
        layer, param_name, param, threshold, expected_sparsity
    )


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "layer,param_name,param,threshold,expected_sparsity",
    [
        (
            Linear(in_features=256, out_features=512),
            "weight",
            torch.randn(512, 256),
            0.0,
            0.0,
        ),
        (
            Linear(in_features=256, out_features=512),
            "weight",
            torch.randn(512, 256),
            1.0,
            0.682,
        ),
        (
            Linear(in_features=256, out_features=512),
            "weight",
            torch.randn(512, 256),
            2.0,
            0.9545,
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            0.0,
            0.0,
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            1.0,
            0.682,
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            2.0,
            0.9545,
        ),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_set_param_mask_from_abs_threshold_cuda(
    layer, param_name, param, threshold, expected_sparsity
):
    layer = layer.to("cuda")
    param = param.to("cuda")
    _test_set_param_mask_from_abs_threshold(
        layer, param_name, param, threshold, expected_sparsity
    )


def _test_set_param_mask_from_sparsity(layer, param_name, param, sparsity):
    mask = ModuleParamKSMask(layer, param_name)
    mask.set_param_data(param)
    mask.set_param_mask_from_sparsity(sparsity)
    measured = tensor_sparsity(mask.param_mask)
    assert (measured - sparsity).abs() < 0.01


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "layer,param_name,param,sparsity",
    [
        (
            Linear(in_features=256, out_features=512),
            "weight",
            torch.randn(512, 256),
            0.0,
        ),
        (
            Linear(in_features=256, out_features=512),
            "weight",
            torch.randn(512, 256),
            0.5,
        ),
        (
            Linear(in_features=256, out_features=512),
            "weight",
            torch.randn(512, 256),
            0.99,
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            0.0,
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            0.5,
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            0.99,
        ),
    ],
)
def test_set_param_mask_from_sparsity(layer, param_name, param, sparsity):
    _test_set_param_mask_from_sparsity(layer, param_name, param, sparsity)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "layer,param_name,param,sparsity",
    [
        (
            Linear(in_features=256, out_features=512),
            "weight",
            torch.randn(512, 256),
            0.0,
        ),
        (
            Linear(in_features=256, out_features=512),
            "weight",
            torch.randn(512, 256),
            0.5,
        ),
        (
            Linear(in_features=256, out_features=512),
            "weight",
            torch.randn(512, 256),
            0.99,
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            0.0,
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            0.5,
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            0.99,
        ),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_set_param_mask_from_sparsity_cuda(layer, param_name, param, sparsity):
    layer = layer.to("cuda")
    param = param.to("cuda")
    _test_set_param_mask_from_sparsity(layer, param_name, param, sparsity)
