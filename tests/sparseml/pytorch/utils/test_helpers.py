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
import tempfile
from typing import Dict, Iterable

import numpy
import pytest
import torch
from torch import Tensor
from torch.nn import BatchNorm2d, Conv2d, Linear, Module, ReLU, Sequential
from torch.optim import SGD
from torch.utils.data import DataLoader

from flaky import flaky
from sparseml.pytorch.datasets import RandNDataset
from sparseml.pytorch.utils import (
    default_device,
    early_stop_data_loader,
    get_optim_learning_rate,
    infinite_data_loader,
    mask_difference,
    set_optim_learning_rate,
    tensor_density,
    tensor_export,
    tensor_sample,
    tensor_sparsity,
    tensors_batch_size,
    tensors_export,
    tensors_module_forward,
    tensors_to_device,
    tensors_to_precision,
    thin_model_from_checkpoint,
)
from tests.sparseml.pytorch.helpers import LinearNet


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_default_device():
    default = default_device()

    if torch.cuda.is_available():
        assert "cuda" in default
    else:
        assert "cpu" in default


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_get_set_optim_learning_rate():
    model = LinearNet()
    optim = SGD(model.parameters(), lr=0.01)

    check_lr = get_optim_learning_rate(optim)
    assert abs(check_lr - 0.01) < 1e-9

    set_optim_learning_rate(optim, 0.0001)

    check_lr = get_optim_learning_rate(optim)
    assert abs(check_lr - 0.0001) < 1e-9


def test_early_stop_data_loader():
    dataset = RandNDataset(100, (3, 32, 32), True)
    data_loader = DataLoader(dataset)

    check_loader = early_stop_data_loader(data_loader, early_stop_steps=10)
    check_count = 0

    for _ in check_loader:
        check_count += 1

    assert check_count == 10

    check_loader = early_stop_data_loader(data_loader, early_stop_steps=-1)
    check_count = 0

    for _ in check_loader:
        check_count += 1

    assert check_count == 100


def test_infinite_data_loader():
    dataset = RandNDataset(100, (3, 32, 32), True)
    data_loader = DataLoader(dataset)

    check_loader = infinite_data_loader(data_loader)
    check_count = 0

    for _ in check_loader:
        check_count += 1

        if check_count >= 150:
            break

    assert check_count == 150


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "tensors,expected",
    [
        (None, -1),
        ([], -1),
        ({}, -1),
        (torch.randn(1, 8, 16, 32), 1),
        (torch.randn(8, 8, 16, 32), 8),
        ((torch.randn(1, 8), torch.randn(8, 8)), 1),
        ([torch.randn(1, 8), torch.randn(8, 8)], 1),
        ({"key": torch.randn(1, 8), "key2": torch.randn(8, 8)}, 1),
        ([[torch.randn(1, 8)], torch.randn(8, 8)], 1),
    ],
)
def test_tensors_batch_size(tensors, expected):
    batch_size = tensors_batch_size(tensors)
    assert batch_size == expected


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "tensors",
    [
        (),
        [],
        {},
        torch.randn(1, 8, 16, 32),
        torch.randn(8, 8, 16, 32),
        (torch.randn(1, 8), torch.randn(8, 8)),
        [torch.randn(1, 8), torch.randn(8, 8)],
        {"key": torch.randn(1, 8), "key2": torch.randn(8, 8)},
        [[torch.randn(1, 8)], torch.randn(8, 8)],
    ],
)
def test_tensors_to_device_cpu(tensors):
    out = tensors_to_device(tensors, "cpu")

    if isinstance(out, Tensor):
        assert not out.is_cuda
    elif isinstance(out, Iterable):
        for tens in out:
            if isinstance(tens, Tensor):
                assert not tens.is_cuda
    elif isinstance(out, Dict):
        for key, tens in out.items():
            if isinstance(tens, Tensor):
                assert not tens.is_cuda


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "tensors",
    [
        (),
        [],
        {},
        torch.randn(1, 8, 16, 32),
        torch.randn(8, 8, 16, 32),
        (torch.randn(1, 8), torch.randn(8, 8)),
        [torch.randn(1, 8), torch.randn(8, 8)],
        {"key": torch.randn(1, 8), "key2": torch.randn(8, 8)},
        [[torch.randn(1, 8)], torch.randn(8, 8)],
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_tensors_to_device_cuda(tensors):
    out = tensors_to_device(tensors, "cuda")

    if isinstance(out, Tensor):
        assert out.is_cuda
    elif isinstance(out, Iterable):
        for tens in out:
            if isinstance(tens, Tensor):
                assert tens.is_cuda
    elif isinstance(out, Dict):
        for key, tens in out.items():
            if isinstance(tens, Tensor):
                assert tens.is_cuda


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "tensors",
    [
        (),
        [],
        {},
        torch.randn(1, 8, 16, 32),
        torch.randn(8, 8, 16, 32),
        (torch.randn(1, 8), torch.randn(8, 8)),
        [torch.randn(1, 8), torch.randn(8, 8)],
        {"key": torch.randn(1, 8), "key2": torch.randn(8, 8)},
        [[torch.randn(1, 8)], torch.randn(8, 8)],
    ],
)
def test_tensors_to_precision_full_cpu(tensors):
    out = tensors_to_precision(tensors, True)

    if isinstance(out, Tensor):
        assert out.dtype == torch.float32
    elif isinstance(out, Iterable):
        for tens in out:
            if isinstance(tens, Tensor):
                assert tens.dtype == torch.float32
    elif isinstance(out, Dict):
        for key, tens in out.items():
            if isinstance(tens, Tensor):
                assert tens.dtype == torch.float32


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "tensors",
    [
        (),
        [],
        {},
        torch.randn(1, 8, 16, 32),
        torch.randn(8, 8, 16, 32),
        (torch.randn(1, 8), torch.randn(8, 8)),
        [torch.randn(1, 8), torch.randn(8, 8)],
        {"key": torch.randn(1, 8), "key2": torch.randn(8, 8)},
        [[torch.randn(1, 8)], torch.randn(8, 8)],
    ],
)
def test_tensors_to_precision_half_cpu(tensors):
    out = tensors_to_precision(tensors, False)

    if isinstance(out, Tensor):
        assert out.dtype == torch.float16
    elif isinstance(out, Iterable):
        for tens in out:
            if isinstance(tens, Tensor):
                assert tens.dtype == torch.float16
    elif isinstance(out, Dict):
        for key, tens in out.items():
            if isinstance(tens, Tensor):
                assert tens.dtype == torch.float16


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "tensors",
    [
        (),
        [],
        {},
        torch.randn(1, 8, 16, 32),
        torch.randn(8, 8, 16, 32),
        (torch.randn(1, 8), torch.randn(8, 8)),
        [torch.randn(1, 8), torch.randn(8, 8)],
        {"key": torch.randn(1, 8), "key2": torch.randn(8, 8)},
        [[torch.randn(1, 8)], torch.randn(8, 8)],
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_tensors_to_precision_full_cuda(tensors):
    tensors = tensors_to_device(tensors, "cuda")
    out = tensors_to_precision(tensors, True)

    if isinstance(out, Tensor):
        assert out.dtype == torch.float32
    elif isinstance(out, Iterable):
        for tens in out:
            if isinstance(tens, Tensor):
                assert tens.dtype == torch.float32
    elif isinstance(out, Dict):
        for key, tens in out.items():
            if isinstance(tens, Tensor):
                assert tens.dtype == torch.float32


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "tensors",
    [
        (),
        [],
        {},
        torch.randn(1, 8, 16, 32),
        torch.randn(8, 8, 16, 32),
        (torch.randn(1, 8), torch.randn(8, 8)),
        [torch.randn(1, 8), torch.randn(8, 8)],
        {"key": torch.randn(1, 8), "key2": torch.randn(8, 8)},
        [[torch.randn(1, 8)], torch.randn(8, 8)],
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_tensors_to_precision_half_cuda(tensors):
    tensors = tensors_to_device(tensors, "cuda")
    out = tensors_to_precision(tensors, False)

    if isinstance(out, Tensor):
        assert out.dtype == torch.float16
    elif isinstance(out, Iterable):
        for tens in out:
            if isinstance(tens, Tensor):
                assert tens.dtype == torch.float16
    elif isinstance(out, Dict):
        for key, tens in out.items():
            if isinstance(tens, Tensor):
                assert tens.dtype == torch.float16


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
class SimpleModule(Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.fc1 = Linear(input_size, 16, bias=True)
        self.relu1 = ReLU()
        self.fc2 = Linear(16, 32, bias=True)
        self.relu2 = ReLU()

    def forward(self, inp):
        out = self.fc1(inp)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)

        return out

    @staticmethod
    def example_input(batch_size: int, input_size: int):
        return torch.randn(batch_size, input_size)

    @staticmethod
    def example_output(batch_size: int):
        return torch.randn(batch_size, 32)


class ComplexModule(Module):
    def __init__(self, input_size_one: int, input_size_two: int):
        super().__init__()
        self.branch1 = Sequential(
            Linear(input_size_one, 16, bias=True), ReLU(), Linear(16, 32), ReLU()
        )
        self.branch2 = Sequential(
            Linear(input_size_two, 16, bias=True),
            ReLU(),
            Linear(16, 32, bias=True),
            ReLU(),
        )
        self.tower = Sequential(Linear(64, 32, bias=True), ReLU())

    def forward(self, inp_one, inp_two):
        out_one = self.branch1(inp_one)
        out_two = self.branch2(inp_two)
        out = torch.cat([out_one, out_two], dim=1)
        out = self.tower(out)

        return out

    @staticmethod
    def example_list_input(batch_size: int, input_size_one: int, input_size_two: int):
        return [
            torch.randn(batch_size, input_size_one),
            torch.randn(batch_size, input_size_two),
        ]

    @staticmethod
    def example_dict_input(batch_size: int, input_size_one: int, input_size_two: int):
        return {
            "inp_one": torch.randn(batch_size, input_size_one),
            "inp_two": torch.randn(batch_size, input_size_two),
        }

    @staticmethod
    def example_output(batch_size: int):
        return torch.randn(batch_size, 32)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "module,tensors,check_feat_lab_inp",
    [
        (SimpleModule(8), SimpleModule.example_input(1, 8), False),
        (SimpleModule(8), SimpleModule.example_input(16, 8), False),
        (ComplexModule(8, 4), ComplexModule.example_list_input(1, 8, 4), False),
        (ComplexModule(8, 4), ComplexModule.example_list_input(16, 8, 4), False),
        (ComplexModule(8, 4), ComplexModule.example_dict_input(1, 8, 4), False),
        (ComplexModule(8, 4), ComplexModule.example_dict_input(16, 8, 4), False),
        (
            SimpleModule(8),
            (SimpleModule.example_input(1, 8), SimpleModule.example_output(1)),
            True,
        ),
        (
            SimpleModule(8),
            [SimpleModule.example_input(16, 8), SimpleModule.example_output(16)],
            True,
        ),
        (
            ComplexModule(8, 4),
            [
                ComplexModule.example_list_input(1, 8, 4),
                ComplexModule.example_output(1),
            ],
            True,
        ),
        (
            ComplexModule(8, 4),
            (
                ComplexModule.example_list_input(16, 8, 4),
                ComplexModule.example_output(16),
            ),
            True,
        ),
        (
            ComplexModule(8, 4),
            (
                ComplexModule.example_dict_input(1, 8, 4),
                ComplexModule.example_output(1),
            ),
            True,
        ),
        (
            ComplexModule(8, 4),
            [
                ComplexModule.example_dict_input(16, 8, 4),
                ComplexModule.example_output(16),
            ],
            True,
        ),
    ],
)
def test_tensors_module_forward(module, tensors, check_feat_lab_inp):
    out = tensors_module_forward(tensors, module, check_feat_lab_inp)
    assert len(out)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "module,tensors,check_feat_lab_inp",
    [
        (SimpleModule(8), SimpleModule.example_input(1, 8), False),
        (SimpleModule(8), SimpleModule.example_input(16, 8), False),
        (ComplexModule(8, 4), ComplexModule.example_list_input(1, 8, 4), False),
        (ComplexModule(8, 4), ComplexModule.example_list_input(16, 8, 4), False),
        (ComplexModule(8, 4), ComplexModule.example_dict_input(1, 8, 4), False),
        (ComplexModule(8, 4), ComplexModule.example_dict_input(16, 8, 4), False),
        (
            SimpleModule(8),
            (SimpleModule.example_input(1, 8), SimpleModule.example_output(1)),
            True,
        ),
        (
            SimpleModule(8),
            [SimpleModule.example_input(16, 8), SimpleModule.example_output(16)],
            True,
        ),
        (
            ComplexModule(8, 4),
            [
                ComplexModule.example_list_input(1, 8, 4),
                ComplexModule.example_output(1),
            ],
            True,
        ),
        (
            ComplexModule(8, 4),
            (
                ComplexModule.example_list_input(16, 8, 4),
                ComplexModule.example_output(16),
            ),
            True,
        ),
        (
            ComplexModule(8, 4),
            (
                ComplexModule.example_dict_input(1, 8, 4),
                ComplexModule.example_output(1),
            ),
            True,
        ),
        (
            ComplexModule(8, 4),
            [
                ComplexModule.example_dict_input(16, 8, 4),
                ComplexModule.example_output(16),
            ],
            True,
        ),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_tensors_module_forward_cuda(module, tensors, check_feat_lab_inp):
    module = module.to("cuda")
    tensors = tensors_to_device(tensors, "cuda")
    out = tensors_module_forward(tensors, module, check_feat_lab_inp)
    assert out is not None


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "tensor,name",
    [
        (torch.randn(1, 8), "small"),
        (torch.randn(16, 32), "larger"),
        (torch.randn(32, 16, 32, 3), "large"),
    ],
)
def test_tensor_export_npy(tensor, name):
    path = tensor_export(tensor, tempfile.gettempdir(), name, npz=False)
    exported = numpy.load(path)

    for s1, s2 in zip(exported.shape, tensor.shape):
        assert s1 == s2


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "tensor,name",
    [
        (torch.randn(1, 8), "small"),
        (torch.randn(16, 32), "larger"),
        (torch.randn(32, 16, 32, 3), "large"),
    ],
)
def test_tensor_export_npz(tensor, name):
    path = tensor_export(tensor, tempfile.gettempdir(), name, npz=True)
    exported = numpy.load(path)
    exported = exported[exported.files[0]]

    for s1, s2 in zip(exported.shape, tensor.shape):
        assert s1 == s2


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "tensor,name",
    [
        (torch.randn(1, 8), "small"),
        (torch.randn(16, 32), "larger"),
        (torch.randn(32, 16, 32, 3), "large"),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_tensor_export_cuda(tensor, name):
    tensor = tensor.to("cuda")
    path = tensor_export(tensor, tempfile.gettempdir(), name)
    exported = numpy.load(path)
    exported = exported[exported.files[0]]

    for s1, s2 in zip(exported.shape, tensor.shape):
        assert s1 == s2


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "tensors,name",
    [
        ((), "empty_tuple"),
        ([], "empty_list"),
        (torch.randn(1, 8, 16, 32), "small_sing_tens"),
        (torch.randn(8, 8, 16, 32), "large_sing_tens"),
        ((torch.randn(1, 8), torch.randn(8, 8)), "flat_tuple"),
        ([torch.randn(1, 8), torch.randn(8, 8)], "flat_list"),
        ([[torch.randn(1, 8)], torch.randn(8, 8)], "nested_list"),
    ],
)
def test_tensors_export(tensors, name):
    paths = tensors_export(tensors, tempfile.gettempdir(), name)

    for path in paths:
        exported = numpy.load(path)
        exported = exported[exported.files[0]]
        assert numpy.sum(exported.shape) > 1


@flaky(max_runs=2, min_passes=1)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "tensor,dim,expected_sparsity",
    [
        (torch.zeros(8, 16), None, torch.tensor(1.0)),
        (torch.zeros(8, 16), 0, torch.ones(8)),
        (torch.zeros(8, 16), 1, torch.ones(16)),
        (torch.zeros(8, 16), [0, 1], torch.ones(8, 16)),
        (torch.zeros(8, 16), [1, 0], torch.ones(16, 8)),
        (torch.zeros(8, 16, 32, 8), [3, 1, 2], torch.ones(8, 16, 32)),
        (torch.ones(8, 16), None, torch.tensor(0.0)),
        (torch.ones(8, 16), 0, torch.zeros(8)),
        (torch.ones(8, 16), 1, torch.zeros(16)),
        (torch.ones(8, 16), [0, 1], torch.zeros(8, 16)),
        (torch.ones(8, 16), [1, 0], torch.zeros(16, 8)),
        (torch.ones(8, 16, 32, 8), [3, 1, 2], torch.zeros(8, 16, 32)),
        (torch.randn(8, 16), None, torch.tensor(0.0)),
        (torch.randn(8, 16), 0, torch.zeros(8)),
        (torch.randn(8, 16), 1, torch.zeros(16)),
        (torch.randn(8, 16), [0, 1], torch.zeros(8, 16)),
        (torch.randn(8, 16), [1, 0], torch.zeros(16, 8)),
        (torch.randn(8, 16, 32, 8), [3, 1, 2], torch.zeros(8, 16, 32)),
        (
            torch.tensor([10.0, 0.0, 1.0, 3.0, 2.0, 0.0, 8.0, 0.0, 5.0, 0.0]),
            None,
            torch.tensor(0.4),
        ),
    ],
)
def test_tensor_sparsity(tensor, dim, expected_sparsity):
    sparsity = tensor_sparsity(tensor, dim)
    assert expected_sparsity.shape == sparsity.shape
    assert torch.sum((sparsity - expected_sparsity).abs()) < 0.001


@flaky(max_runs=2, min_passes=1)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "tensor,dim,expected_sparsity",
    [
        (torch.zeros(8, 16), None, torch.tensor(1.0)),
        (torch.zeros(8, 16), 0, torch.ones(8)),
        (torch.zeros(8, 16, 32, 8), [3, 1, 2], torch.ones(8, 16, 32)),
        (torch.ones(8, 16), None, torch.tensor(0.0)),
        (torch.ones(8, 16), 0, torch.zeros(8)),
        (torch.ones(8, 16, 32, 8), [3, 1, 2], torch.zeros(8, 16, 32)),
        (torch.randn(8, 16), None, torch.tensor(0.0)),
        (torch.randn(8, 16), 0, torch.zeros(8)),
        (torch.randn(8, 16, 32, 8), [3, 1, 2], torch.zeros(8, 16, 32)),
        (
            torch.tensor([10.0, 0.0, 1.0, 3.0, 2.0, 0.0, 8.0, 0.0, 5.0, 0.0]),
            None,
            torch.tensor(0.4),
        ),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_tensor_sparsity_cuda(tensor, dim, expected_sparsity):
    tensor = tensor.to("cuda")
    sparsity = tensor_sparsity(tensor, dim)
    assert expected_sparsity.shape == sparsity.shape
    assert torch.sum((sparsity.detach().cpu() - expected_sparsity).abs()) < 0.001


@flaky(max_runs=2, min_passes=1)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "tensor,dim,expected_density",
    [
        (torch.zeros(8, 16), None, torch.tensor(0.0)),
        (torch.zeros(8, 16), 0, torch.zeros(8)),
        (torch.zeros(8, 16), 1, torch.zeros(16)),
        (torch.zeros(8, 16), [0, 1], torch.zeros(8, 16)),
        (torch.zeros(8, 16), [1, 0], torch.zeros(16, 8)),
        (torch.zeros(8, 16, 32, 8), [3, 1, 2], torch.zeros(8, 16, 32)),
        (torch.ones(8, 16), None, torch.tensor(1.0)),
        (torch.ones(8, 16), 0, torch.ones(8)),
        (torch.ones(8, 16), 1, torch.ones(16)),
        (torch.ones(8, 16), [0, 1], torch.ones(8, 16)),
        (torch.ones(8, 16), [1, 0], torch.ones(16, 8)),
        (torch.ones(8, 16, 32, 8), [3, 1, 2], torch.ones(8, 16, 32)),
        (torch.randn(8, 16), None, torch.tensor(1.0)),
        (torch.randn(8, 16), 0, torch.ones(8)),
        (torch.randn(8, 16), 1, torch.ones(16)),
        (torch.randn(8, 16), [0, 1], torch.ones(8, 16)),
        (torch.randn(8, 16), [1, 0], torch.ones(16, 8)),
        (torch.randn(8, 16, 32, 8), [3, 1, 2], torch.ones(8, 16, 32)),
        (
            torch.tensor([10.0, 0.0, 1.0, 3.0, 2.0, 0.0, 8.0, 0.0, 5.0, 0.0]),
            None,
            torch.tensor(0.6),
        ),
    ],
)
def test_tensor_density(tensor, dim, expected_density):
    density = tensor_density(tensor, dim)
    assert expected_density.shape == density.shape
    assert torch.sum((density - expected_density).abs()) < 0.001


@flaky(max_runs=2, min_passes=1)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "tensor,dim,expected_density",
    [
        (torch.zeros(8, 16), None, torch.tensor(0.0)),
        (torch.zeros(8, 16, 32, 8), [3, 1, 2], torch.zeros(8, 16, 32)),
        (torch.ones(8, 16), None, torch.tensor(1.0)),
        (torch.ones(8, 16, 32, 8), [3, 1, 2], torch.ones(8, 16, 32)),
        (torch.randn(8, 16), None, torch.tensor(1.0)),
        (
            torch.tensor([10.0, 0.0, 1.0, 3.0, 2.0, 0.0, 8.0, 0.0, 5.0, 0.0]),
            None,
            torch.tensor(0.6),
        ),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_tensor_density_cuda(tensor, dim, expected_density):
    tensor = tensor.to("cuda")
    density = tensor_density(tensor, dim)
    assert expected_density.shape == density.shape
    assert torch.sum((density.detach().cpu() - expected_density).abs()) < 0.001


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "tensor,size,dim,expected_shape",
    [
        (torch.randn(8, 16), 100, None, [100]),
        (torch.randn(8, 16), 100, 0, [8, 100]),
        (torch.randn(8, 16), 100, 1, [16, 100]),
        (torch.randn(8, 16), 10, [0, 1], [8, 16, 10]),
        (torch.randn(8, 16), 10, [1, 0], [16, 8, 10]),
        (torch.randn(64, 12, 32, 16), 10, 2, [32, 10]),
        (torch.randn(64, 12, 32, 16), 10, [3, 2], [16, 32, 10]),
        (torch.randn(64, 12, 32, 16), 10, 1, [12, 10]),
        (torch.randn(64, 12, 32, 16), 10, [0, 1], [64, 12, 10]),
    ],
)
def test_tensor_sample(tensor, size, dim, expected_shape):
    sample = tensor_sample(tensor, size, dim)
    assert len(sample.shape) == len(expected_shape)
    for s1, s2 in zip(sample.shape, expected_shape):
        assert s1 == s2


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "tensor,size,dim,expected_shape",
    [
        (torch.randn(8, 16), 100, None, [100]),
        (torch.randn(8, 16), 100, 0, [8, 100]),
        (torch.randn(8, 16), 100, 1, [16, 100]),
        (torch.randn(8, 16), 10, [0, 1], [8, 16, 10]),
        (torch.randn(8, 16), 10, [1, 0], [16, 8, 10]),
        (torch.randn(64, 12, 32, 16), 10, 2, [32, 10]),
        (torch.randn(64, 12, 32, 16), 10, [3, 2], [16, 32, 10]),
        (torch.randn(64, 12, 32, 16), 10, 1, [12, 10]),
        (torch.randn(64, 12, 32, 16), 10, [0, 1], [64, 12, 10]),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_tensor_sample_cuda(tensor, size, dim, expected_shape):
    tensor = tensor.to("cuda")
    sample = tensor_sample(tensor, size, dim)
    assert len(sample.shape) == len(expected_shape)
    for s1, s2 in zip(sample.shape, expected_shape):
        assert s1 == s2


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "old_mask,new_mask,expected_diff",
    [
        (torch.zeros(8, 8), torch.zeros(8, 8), torch.zeros(8, 8)),
        (torch.zeros(8, 8), torch.ones(8, 8), torch.ones(8, 8)),
        (torch.ones(8, 8), torch.zeros(8, 8), -1.0 * torch.ones(8, 8)),
        (torch.ones(8, 8), torch.ones(8, 8), torch.zeros(8, 8)),
        (
            torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 1.0]),
            torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            torch.tensor([0.0, 1.0, -1.0, 0.0, -1.0, 0.0]),
        ),
    ],
)
def test_mask_difference(old_mask, new_mask, expected_diff):
    diff = mask_difference(old_mask, new_mask)
    assert torch.sum((diff - expected_diff).abs()) < sys.float_info.epsilon


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "model,state_dict,test_input",
    [
        (
            Sequential(Conv2d(3, 16, (1, 1)), BatchNorm2d(16), Conv2d(16, 16, (1, 1))),
            {
                "0.weight": torch.randn(8, 3, 1, 1),
                "0.bias": torch.randn(8),
                "1.weight": torch.randn(8),
                "1.bias": torch.randn(8),
                "1.running_mean": torch.randn(8),
                "1.running_var": torch.randn(8),
                "2.weight": torch.randn(12, 8, 1, 1),
                "2.bias": torch.randn(12),
            },
            torch.randn(2, 3, 16, 16),
        ),
        (
            Sequential(Linear(8, 12), Linear(12, 16)),
            {
                "0.weight": torch.randn(7, 8),
                "0.bias": torch.randn(7),
                "1.weight": torch.randn(9, 7),
                "1.bias": torch.randn(9),
            },
            torch.randn(5, 8),
        ),
    ],
)
def test_thin_model_from_checkpoint(model, state_dict, test_input):
    with pytest.raises(RuntimeError):
        model.load_state_dict(state_dict)

    thin_model_from_checkpoint(model, state_dict)
    model.load_state_dict(state_dict, strict=True)
    assert isinstance(model(test_input), Tensor)
