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
import sys

import pytest
import torch
from torch.nn import Conv2d, Linear

from sparseml.pytorch.optim import (
    BlockPruningMaskCreator,
    DimensionSparsityMaskCreator,
    GroupedPruningMaskCreator,
    ModuleParamPruningMask,
    UnstructuredPruningMaskCreator,
)
from sparseml.pytorch.utils import tensor_sparsity


def _test_constructor(layer, param_name, mask_creator):
    mask = ModuleParamPruningMask([layer], [param_name], mask_creator=mask_creator)
    assert mask.layers[0] == layer
    assert mask.param_names[0] == param_name
    assert not mask.store_init
    assert not mask.store_unmasked
    assert mask.track_grad_mom == -1.0
    assert not mask.global_sparsity
    assert not mask.enabled
    assert mask_creator == mask.mask_creator


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "layer,param_name,mask_creator",
    [
        (
            Linear(in_features=8, out_features=64),
            "weight",
            UnstructuredPruningMaskCreator(),
        ),
        (
            Linear(in_features=8, out_features=64),
            "bias",
            UnstructuredPruningMaskCreator(),
        ),
        (
            Linear(in_features=8, out_features=64),
            "weight",
            DimensionSparsityMaskCreator(1),
        ),
        (
            Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            "weight",
            UnstructuredPruningMaskCreator(),
        ),
        (
            Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            "bias",
            UnstructuredPruningMaskCreator(),
        ),
        (
            Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            "weight",
            BlockPruningMaskCreator([4, 1]),
        ),
    ],
)
def test_constructor(layer, param_name, mask_creator):
    _test_constructor(layer, param_name, mask_creator=mask_creator)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "layer,param_name, mask_creator",
    [
        (
            Linear(in_features=8, out_features=64),
            "weight",
            UnstructuredPruningMaskCreator(),
        ),
        (
            Linear(in_features=8, out_features=64),
            "weight",
            DimensionSparsityMaskCreator(1),
        ),
        (
            Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            "weight",
            UnstructuredPruningMaskCreator(),
        ),
        (
            Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            "weight",
            BlockPruningMaskCreator([4, 1]),
        ),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_constructor_cuda(layer, param_name, mask_creator):
    layer = layer.to("cuda")
    _test_constructor(layer, param_name, mask_creator=mask_creator)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def _test_set_param_data(layer, param_name, data):
    mask = ModuleParamPruningMask([layer], param_name)
    mask.set_param_data(data, 0)
    assert torch.sum((mask.params_data[0] - data).abs()) < sys.float_info.epsilon


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
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
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
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
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def _test_set_param_mask(layer, param_name, param_mask):
    mask = ModuleParamPruningMask([layer], [param_name])
    result = mask.set_param_masks([param_mask])[0]
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
    param_data_zeros = (mask.params_data[0] == 0.0).type("float32")

    assert torch.sum((param_data_zeros - mask_zeros).abs()) < sys.float_info.epsilon


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
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
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
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


def _test_grouped_sparsity_mask_output(mask_creator, mask):
    grouped_mask = mask_creator.group_tensor(mask)
    mask_vals_are_grouped = torch.all((grouped_mask == 0.0) | (grouped_mask == 1.0))
    assert mask_vals_are_grouped


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def _test_set_param_mask_from_abs_threshold(
    layer,
    param_name,
    param,
    threshold,
    expected_sparsity,
    mask_creator,
):
    mask = ModuleParamPruningMask([layer], [param_name], mask_creator=mask_creator)
    mask.set_param_data(param, 0)
    mask.set_param_masks_from_abs_threshold(threshold)
    sparsity = tensor_sparsity(mask.param_masks[0])
    assert (sparsity - expected_sparsity).abs() < 0.01
    if isinstance(mask_creator, GroupedPruningMaskCreator):
        _test_grouped_sparsity_mask_output(mask_creator, mask.param_masks[0])


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "layer,param_name,param,threshold,expected_sparsity,mask_creator",
    [
        (
            Linear(in_features=256, out_features=512),
            "weight",
            torch.randn(512, 256),
            0.0,
            0.0,
            UnstructuredPruningMaskCreator(),
        ),
        (
            Linear(in_features=256, out_features=512),
            "weight",
            torch.randn(512, 256),
            1.0,
            0.682,
            UnstructuredPruningMaskCreator(),
        ),
        (
            Linear(in_features=256, out_features=512),
            "weight",
            torch.randn(512, 256),
            2.0,
            0.9545,
            UnstructuredPruningMaskCreator(),
        ),
        (
            Linear(in_features=256 * 256, out_features=32),
            "weight",
            torch.randn(32, 256 * 256),
            0.8,
            0.5188,
            DimensionSparsityMaskCreator(1),
        ),
        (
            Linear(in_features=256 * 256, out_features=32),
            "weight",
            torch.randn(32, 256 * 256),
            1.5,
            0.9809,
            BlockPruningMaskCreator([1, 4]),
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            0.0,
            0.0,
            UnstructuredPruningMaskCreator(),
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            1.0,
            0.682,
            UnstructuredPruningMaskCreator(),
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            2.0,
            0.9545,
            UnstructuredPruningMaskCreator(),
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            1.0,
            0.8428,
            DimensionSparsityMaskCreator([0, 1]),
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            2.0,
            0.9995,
            BlockPruningMaskCreator([1, 4]),
        ),
    ],
)
def test_set_param_mask_from_abs_threshold(
    layer,
    param_name,
    param,
    threshold,
    expected_sparsity,
    mask_creator,
):
    _test_set_param_mask_from_abs_threshold(
        layer, param_name, param, threshold, expected_sparsity, mask_creator
    )


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "layer,param_name,param,threshold,expected_sparsity, mask_creator",
    [
        (
            Linear(in_features=256, out_features=512),
            "weight",
            torch.randn(512, 256),
            0.0,
            0.0,
            UnstructuredPruningMaskCreator(),
        ),
        (
            Linear(in_features=256, out_features=512),
            "weight",
            torch.randn(512, 256),
            1.0,
            0.682,
            UnstructuredPruningMaskCreator(),
        ),
        (
            Linear(in_features=256, out_features=512),
            "weight",
            torch.randn(512, 256),
            2.0,
            0.9545,
            UnstructuredPruningMaskCreator(),
        ),
        (
            Linear(in_features=256 * 256, out_features=32),
            "weight",
            torch.randn(32, 256 * 256),
            0.8,
            0.5188,
            DimensionSparsityMaskCreator(1),
        ),
        (
            Linear(in_features=256 * 256, out_features=32),
            "weight",
            torch.randn(32, 256 * 256),
            1.5,
            0.9809,
            BlockPruningMaskCreator([1, 4]),
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            0.0,
            0.0,
            UnstructuredPruningMaskCreator(),
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            1.0,
            0.682,
            UnstructuredPruningMaskCreator(),
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            2.0,
            0.9545,
            UnstructuredPruningMaskCreator(),
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            1.0,
            0.8428,
            DimensionSparsityMaskCreator([0, 1]),
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            2.0,
            0.9995,
            BlockPruningMaskCreator([1, 4]),
        ),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_set_param_mask_from_abs_threshold_cuda(
    layer, param_name, param, threshold, expected_sparsity, mask_creator
):
    layer = layer.to("cuda")
    param = param.to("cuda")
    _test_set_param_mask_from_abs_threshold(
        layer, param_name, param, threshold, expected_sparsity, mask_creator
    )


def _test_set_param_mask_from_sparsity(
    layer, param_name, param, sparsity, mask_creator
):
    mask = ModuleParamPruningMask([layer], [param_name], mask_creator=mask_creator)
    mask.set_param_data(param, 0)
    mask.set_param_masks_from_sparsity(sparsity)
    measured = tensor_sparsity(mask.param_masks[0])
    assert (measured - sparsity).abs() < 0.01
    if isinstance(mask_creator, GroupedPruningMaskCreator):
        _test_grouped_sparsity_mask_output(mask_creator, mask.param_masks[0])


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "layer,param_name,param,sparsity,mask_creator",
    [
        (
            Linear(in_features=256, out_features=512),
            "weight",
            torch.randn(512, 256),
            0.0,
            UnstructuredPruningMaskCreator(),
        ),
        (
            Linear(in_features=256, out_features=512),
            "weight",
            torch.randn(512, 256),
            0.5,
            UnstructuredPruningMaskCreator(),
        ),
        (
            Linear(in_features=256, out_features=512),
            "weight",
            torch.randn(512, 256),
            0.99,
            UnstructuredPruningMaskCreator(),
        ),
        (
            Linear(in_features=256, out_features=512),
            "weight",
            torch.randn(512, 256),
            0.99,
            DimensionSparsityMaskCreator(1),
        ),
        (
            Linear(in_features=256, out_features=512),
            "weight",
            torch.randn(512, 256),
            0.6,
            BlockPruningMaskCreator([1, 4]),
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            0.0,
            UnstructuredPruningMaskCreator(),
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            0.5,
            UnstructuredPruningMaskCreator(),
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            0.99,
            UnstructuredPruningMaskCreator(),
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            0.6,
            DimensionSparsityMaskCreator([0, 1]),
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            0.99,
            BlockPruningMaskCreator([1, 4]),
        ),
    ],
)
def test_set_param_mask_from_sparsity(layer, param_name, param, sparsity, mask_creator):
    _test_set_param_mask_from_sparsity(layer, param_name, param, sparsity, mask_creator)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "layer,param_name,param,sparsity,mask_creator",
    [
        (
            Linear(in_features=256, out_features=512),
            "weight",
            torch.randn(512, 256),
            0.0,
            UnstructuredPruningMaskCreator(),
        ),
        (
            Linear(in_features=256, out_features=512),
            "weight",
            torch.randn(512, 256),
            0.5,
            UnstructuredPruningMaskCreator(),
        ),
        (
            Linear(in_features=256, out_features=512),
            "weight",
            torch.randn(512, 256),
            0.99,
            UnstructuredPruningMaskCreator(),
        ),
        (
            Linear(in_features=256, out_features=512),
            "weight",
            torch.randn(512, 256),
            0.99,
            DimensionSparsityMaskCreator(1),
        ),
        (
            Linear(in_features=256, out_features=512),
            "weight",
            torch.randn(512, 256),
            0.6,
            BlockPruningMaskCreator([1, 4]),
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            0.0,
            UnstructuredPruningMaskCreator(),
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            0.5,
            UnstructuredPruningMaskCreator(),
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            0.99,
            UnstructuredPruningMaskCreator(),
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            0.6,
            DimensionSparsityMaskCreator([0, 1]),
        ),
        (
            Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            "weight",
            torch.randn(512, 256, 3, 3),
            0.99,
            BlockPruningMaskCreator([1, 4]),
        ),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_set_param_mask_from_sparsity_cuda(
    layer, param_name, param, sparsity, mask_creator
):
    layer = layer.to("cuda")
    param = param.to("cuda")
    _test_set_param_mask_from_sparsity(layer, param_name, param, sparsity, mask_creator)
