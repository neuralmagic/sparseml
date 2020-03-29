import pytest

from typing import Iterable, Dict
import sys
import tempfile
import numpy
import torch
from torch import Tensor
from torch.nn import Module, Sequential, Linear, ReLU

from neuralmagicML.pytorch.utils import (
    tensors_batch_size,
    tensors_to_device,
    tensors_to_precision,
    tensors_module_forward,
    tensor_export,
    tensors_export,
    tensor_density,
    tensor_sparsity,
    tensor_sample,
    abs_threshold_from_sparsity,
    sparsity_mask_from_abs_threshold,
    sparsity_mask,
    sparsity_mask_from_tensor,
    mask_difference,
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


@pytest.mark.parametrize(
    "tensor,name",
    [
        (torch.randn(1, 8), "small"),
        (torch.randn(16, 32), "larger"),
        (torch.randn(32, 16, 32, 3), "large"),
    ],
)
def test_tensor_export_npy(tensor, name):
    path = tensor_export(tensor, tempfile.gettempdir(), name, npz=True)
    exported = numpy.load(path)
    exported = exported[exported.files[0]]

    for s1, s2 in zip(exported.shape, tensor.shape):
        assert s1 == s2


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


@pytest.mark.parametrize(
    "tensor,sparsity,expected_thresh",
    [
        (torch.tensor([]), 0.0, None),
        (torch.zeros(8, 8), 0.0, None),
        (torch.zeros(8, 8), 0.5, 0.0),
        (torch.zeros(8, 8), 1.0, 0.0),
        (torch.ones(8, 8), 0.0, None),
        (torch.ones(8, 8), 0.5, 1.0),
        (torch.ones(8, 8), 1.0, 1.0),
        (torch.tensor([2.0, 1.0, 7.0, 4.0, 6.0, 5.0, 3.0, 8.0, 10.0]), 0.0, None),
        (torch.tensor([2.0, 1.0, 7.0, 4.0, 6.0, 5.0, 3.0, 8.0, 10.0]), 0.2, 3.0),
        (torch.tensor([2.0, 1.0, 7.0, 4.0, 6.0, 5.0, 3.0, 8.0, 10.0]), 0.5, 5.0),
        (torch.tensor([2.0, 1.0, 7.0, 4.0, 6.0, 5.0, 3.0, 8.0, 10.0]), 0.8, 7.0),
        (torch.tensor([2.0, 1.0, 7.0, 4.0, 6.0, 5.0, 3.0, 8.0, 10.0]), 1.0, 10.0),
    ],
)
def test_abs_threshold_from_sparsity(tensor, sparsity, expected_thresh):
    thresh = abs_threshold_from_sparsity(tensor, sparsity)

    if expected_thresh is not None:
        assert abs(thresh - expected_thresh) < 0.001
    else:
        assert numpy.sum(thresh.shape) < 1


@pytest.mark.parametrize(
    "tensor,sparsity,expected_thresh",
    [
        (torch.tensor([]), 0.0, None),
        (torch.zeros(8, 8), 0.0, None),
        (torch.ones(8, 8), 0.0, None),
        (torch.tensor([2.0, 1.0, 7.0, 4.0, 6.0, 5.0, 3.0, 8.0, 10.0]), 0.0, None),
        (torch.tensor([2.0, 1.0, 7.0, 4.0, 6.0, 5.0, 3.0, 8.0, 10.0]), 0.5, 5.0),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_abs_threshold_from_sparsity_cuda(tensor, sparsity, expected_thresh):
    tensor = tensor.to("cuda")
    thresh = abs_threshold_from_sparsity(tensor, sparsity)

    if expected_thresh is not None:
        assert abs(thresh.detach().cpu() - expected_thresh) < 0.001
    else:
        assert numpy.sum(thresh.shape) < 1


@pytest.mark.parametrize(
    "tensor,sparsity,expected_mask",
    [
        (torch.tensor([]), 0.0, torch.tensor([])),
        (torch.zeros(8, 8), 1.0, torch.zeros(8, 8)),
        (torch.zeros(8, 8), 0.0, torch.ones(8, 8)),
        (torch.ones(8, 8), 1.0, torch.zeros(8, 8)),
        (torch.ones(8, 8), 0.0, torch.ones(8, 8)),
        (
            torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
            0.0,
            torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
        ),
        (
            torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
            0.2,
            torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]),
        ),
        (
            torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
            0.5,
            torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]),
        ),
        (
            torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
            0.8,
            torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
        ),
        (
            torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
            1.0,
            torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]),
        ),
    ],
)
def test_sparsity_mask(tensor, sparsity, expected_mask):
    mask = sparsity_mask(tensor, sparsity)
    assert torch.sum((mask - expected_mask).abs()) < 0.001


@pytest.mark.parametrize(
    "tensor,sparsity,expected_mask",
    [
        (torch.tensor([]), 0.0, torch.tensor([])),
        (torch.zeros(8, 8), 1.0, torch.zeros(8, 8)),
        (torch.zeros(8, 8), 0.0, torch.ones(8, 8)),
        (torch.ones(8, 8), 1.0, torch.zeros(8, 8)),
        (torch.ones(8, 8), 0.0, torch.ones(8, 8)),
        (
            torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
            0.0,
            torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
        ),
        (
            torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
            0.2,
            torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]),
        ),
        (
            torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
            0.5,
            torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]),
        ),
        (
            torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
            0.8,
            torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
        ),
        (
            torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
            1.0,
            torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]),
        ),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_sparsity_mask_cuda(tensor, sparsity, expected_mask):
    mask = sparsity_mask(tensor, sparsity)
    assert torch.sum((mask - expected_mask).abs()) < 0.001


@pytest.mark.parametrize(
    "tensor,expected_mask",
    [
        (torch.zeros(8, 8), torch.zeros(8, 8)),
        (torch.ones(8, 8), torch.ones(8, 8)),
        (torch.randn(8, 8), torch.ones(8, 8)),
        (
            torch.tensor([1.0, 0.0, 3.0, 0.0, 0.0, 6.0, 7.0, 0.0]),
            torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]),
        ),
    ],
)
def test_sparsity_mask_from_tensor(tensor: Tensor, expected_mask):
    mask = sparsity_mask_from_tensor(tensor)
    assert torch.sum((mask - expected_mask).abs()) < 0.001


@pytest.mark.parametrize(
    "tensor,expected_mask",
    [
        (torch.zeros(8, 8), torch.zeros(8, 8)),
        (torch.ones(8, 8), torch.ones(8, 8)),
        (torch.randn(8, 8), torch.ones(8, 8)),
        (
            torch.tensor([1.0, 0.0, 3.0, 0.0, 0.0, 6.0, 7.0, 0.0]),
            torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]),
        ),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_sparsity_mask_from_tensor_cuda(tensor: Tensor, expected_mask):
    tensor = tensor.to("cuda")
    mask = sparsity_mask_from_tensor(tensor)
    assert torch.sum((mask.detach().cpu() - expected_mask).abs()) < 0.001


@pytest.mark.parametrize(
    "tensor,threshold,expected_mask",
    [
        (torch.tensor([]), 0.0, torch.tensor([])),
        (torch.zeros(8, 8), 1.0, torch.zeros(8, 8)),
        (torch.zeros(8, 8), 0.0, torch.zeros(8, 8)),
        (torch.ones(8, 8), 1.0, torch.zeros(8, 8)),
        (torch.ones(8, 8), 0.5, torch.ones(8, 8)),
        (torch.ones(8, 8), 2.0, torch.zeros(8, 8)),
        (
            torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
            0.0,
            torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0]),
        ),
        (
            torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
            1.0,
            torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]),
        ),
        (
            torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
            2.0,
            torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]),
        ),
        (
            torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
            3.0,
            torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
        ),
        (
            torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
            4.0,
            torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]),
        ),
    ],
)
def test_sparsity_mask_from_abs_threshold(tensor, threshold, expected_mask):
    mask = sparsity_mask_from_abs_threshold(tensor, threshold)
    assert torch.sum(mask - expected_mask).abs() < 0.001


@pytest.mark.parametrize(
    "tensor,threshold,expected_mask",
    [
        (torch.tensor([]), 0.0, torch.tensor([])),
        (torch.zeros(8, 8), 1.0, torch.zeros(8, 8)),
        (torch.zeros(8, 8), 0.0, torch.zeros(8, 8)),
        (torch.ones(8, 8), 1.0, torch.zeros(8, 8)),
        (torch.ones(8, 8), 0.5, torch.ones(8, 8)),
        (torch.ones(8, 8), 2.0, torch.zeros(8, 8)),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_sparsity_mask_from_abs_threshold_cuda(tensor, threshold, expected_mask):
    tensor = tensor.to("cuda")
    mask = sparsity_mask_from_abs_threshold(tensor, threshold)
    assert torch.sum(mask.detach().cpu() - expected_mask).abs() < 0.001


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
