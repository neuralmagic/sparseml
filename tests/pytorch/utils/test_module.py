import pytest

import math
import torch
import torch.nn.functional as TF
from torch.nn import Sequential, Linear, ReLU
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam

from neuralmagicML.pytorch.utils import (
    DEFAULT_LOSS_KEY,
    def_model_backward,
    ModuleRunHooks,
    ModuleRunFuncs,
    ModuleRunResults,
    ModuleTrainer,
    ModuleTester,
    tensors_batch_size,
    tensors_to_device,
    tensors_module_forward,
    LossWrapper,
)


def default_calcs_for_backwards():
    x = torch.ones(2, 2, requires_grad=True)
    y = x + 2
    z = y * y * 3
    out = z.mean()

    return out


def test_def_model_backward():
    losses = {DEFAULT_LOSS_KEY: default_calcs_for_backwards()}
    module = Sequential(Linear(8, 8), ReLU())
    def_model_backward(losses, module)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_def_model_backward_cuda():
    def _run_calcs():
        x = torch.ones(2, 2, requires_grad=True).to("cuda")
        y = x + 2
        z = y * y * 3
        out = z.mean()

        return out

    losses = {DEFAULT_LOSS_KEY: _run_calcs()}
    module = Sequential(Linear(8, 8), ReLU()).to("cuda")
    def_model_backward(losses, module)


def test_run_hooks_batch_start():
    call_counter = 0

    def hook(counter, step_count, batch_size, data):
        nonlocal call_counter
        call_counter += 1

    hooks = ModuleRunHooks()
    ref = hooks.register_batch_start_hook(hook)
    assert call_counter == 0
    hooks.invoke_batch_start(0, 0, 1, None)
    assert call_counter == 1
    ref.remove()
    hooks.invoke_batch_start(0, 0, 1, None)
    assert call_counter == 1


def test_run_hooks_batch_forward():
    call_counter = 0

    def hook(counter, step_count, batch_size, data, pred):
        nonlocal call_counter
        call_counter += 1

    hooks = ModuleRunHooks()
    ref = hooks.register_batch_forward_hook(hook)
    assert call_counter == 0
    hooks.invoke_batch_forward(0, 0, 1, None, None)
    assert call_counter == 1
    ref.remove()
    hooks.invoke_batch_forward(0, 0, 1, None, None)
    assert call_counter == 1


def test_run_hooks_batch_loss():
    call_counter = 0

    def hook(counter, step_count, batch_size, data, pred, losses):
        nonlocal call_counter
        call_counter += 1

    hooks = ModuleRunHooks()
    ref = hooks.register_batch_loss_hook(hook)
    assert call_counter == 0
    hooks.invoke_batch_loss(0, 0, 1, None, None, {})
    assert call_counter == 1
    ref.remove()
    hooks.invoke_batch_loss(0, 0, 1, None, None, {})
    assert call_counter == 1


def test_run_hooks_batch_backward():
    call_counter = 0

    def hook(counter, step_count, batch_size, data, pred, losses):
        nonlocal call_counter
        call_counter += 1

    hooks = ModuleRunHooks()
    ref = hooks.register_batch_backward_hook(hook)
    assert call_counter == 0
    hooks.invoke_batch_backward(0, 0, 1, None, None, {})
    assert call_counter == 1
    ref.remove()
    hooks.invoke_batch_backward(0, 0, 1, None, None, {})
    assert call_counter == 1


def test_run_hooks_batch_end():
    call_counter = 0

    def hook(counter, step_count, batch_size, data, pred, losses):
        nonlocal call_counter
        call_counter += 1

    hooks = ModuleRunHooks()
    ref = hooks.register_batch_end_hook(hook)
    assert call_counter == 0
    hooks.invoke_batch_end(0, 0, 1, None, None, {})
    assert call_counter == 1
    ref.remove()
    hooks.invoke_batch_end(0, 0, 1, None, None, {})
    assert call_counter == 1


def test_run_funcs_batch_size():
    funcs = ModuleRunFuncs()
    assert funcs.batch_size.__name__ == tensors_batch_size.__name__
    assert funcs.batch_size(torch.randn(8, 4)) == 8


def test_run_funcs_to_device():
    funcs = ModuleRunFuncs()
    assert funcs.to_device.__name__ == tensors_to_device.__name__
    assert not funcs.to_device(torch.randn(8, 4), "cpu").is_cuda


def test_run_funcs_model_forward():
    funcs = ModuleRunFuncs()
    assert funcs.model_forward.__name__ == tensors_module_forward.__name__
    out = funcs.model_forward(torch.randn(8, 4), Sequential(Linear(4, 8), ReLU()))
    assert out.shape[0] == 8
    assert out.shape[1] == 8


def test_run_funcs_model_backward():
    funcs = ModuleRunFuncs()
    assert funcs.model_backward.__name__ == def_model_backward.__name__
    losses = {DEFAULT_LOSS_KEY: default_calcs_for_backwards()}
    module = Sequential(Linear(8, 8), ReLU())
    funcs.model_backward(losses, module)


def test_run_funcs_copy():
    funcs = ModuleRunFuncs()
    funcs.batch_size = None
    funcs.to_device = None
    funcs.model_forward = None
    funcs.model_backward = None

    copy_funcs = ModuleRunFuncs()
    copy_funcs.copy(funcs)
    assert copy_funcs.batch_size is None
    assert copy_funcs.to_device is None
    assert copy_funcs.model_forward is None
    assert copy_funcs.model_backward is None


@pytest.mark.parametrize(
    "name,loss_tensors,batch_size,expected_mean,expected_std",
    [
        ("zeros", [torch.tensor(0.0)], 100, 0.0, 0.0),
        ("ones", [torch.tensor(1.0)], 100, 1.0, 0.0),
        (
            "mixed",
            [torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0)],
            20,
            1.0 / 3.0,
            0.4754,
        ),
    ],
)
def test_run_results(name, loss_tensors, batch_size, expected_mean, expected_std):
    results = ModuleRunResults()

    for loss in loss_tensors:
        results.append({name: loss}, batch_size)

    mean = results.result_mean(name)
    std = results.result_std(name)

    assert len(results.result(name)) == len(loss_tensors)
    assert (mean - expected_mean).abs() < 0.0001
    assert (std - expected_std).abs() < 0.0001


TEST_MODULE = Sequential(
    Linear(8, 16), ReLU(), Linear(16, 32), ReLU(), Linear(32, 1), ReLU()
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


def _train_helper(model, dataset, loss, optimizer, batch_size, device):
    counters = {"start": 0, "forward": 0, "loss": 0, "backward": 0, "end": 0}

    def batch_start_hook(counter, step_count, batch_size, data):
        nonlocal counters
        counters["start"] += 1

    def batch_forward_hook(counter, step_count, batch_size, data, pred):
        nonlocal counters
        counters["forward"] += 1

    def batch_loss_hook(counter, step_count, batch_size, data, pred, losses):
        nonlocal counters
        counters["loss"] += 1

    def batch_backward_hook(counter, step_count, batch_size, data, pred, losses):
        nonlocal counters
        counters["backward"] += 1

    def batch_end_hook(counter, step_count, batch_size, data, pred, losses):
        nonlocal counters
        counters["end"] += 1

    data_loader = DataLoader(dataset, batch_size)
    trainer = ModuleTrainer(model, device, loss, optimizer)
    hook_refs = [
        trainer.run_hooks.register_batch_start_hook(batch_start_hook),
        trainer.run_hooks.register_batch_forward_hook(batch_forward_hook),
        trainer.run_hooks.register_batch_loss_hook(batch_loss_hook),
        trainer.run_hooks.register_batch_backward_hook(batch_backward_hook),
        trainer.run_hooks.register_batch_end_hook(batch_end_hook),
    ]
    result = trainer.run(data_loader, desc="", track_results=False)
    assert not result

    for ref in hook_refs:
        ref.remove()

    expected_batches = math.ceil(len(dataset) / float(batch_size))
    for key, call_count in counters.items():
        assert expected_batches == call_count

    result = trainer.run_epoch(
        data_loader, epoch=0, show_progress=False, track_results=True
    )
    assert isinstance(result, ModuleRunResults)

    for key, losses in result.results.items():
        assert len(losses) == expected_batches

        for loss in losses:
            assert not loss.is_cuda


@pytest.mark.parametrize(
    "model,dataset,loss,optimizer,batch_size",
    [
        (
            TEST_MODULE,
            TestingDataset(100),
            LossWrapper(TF.mse_loss),
            SGD(TEST_MODULE.parameters(), 0.001),
            16,
        ),
        (
            TEST_MODULE,
            TestingDataset(100),
            LossWrapper(TF.mse_loss),
            Adam(TEST_MODULE.parameters()),
            27,
        ),
    ],
)
def test_module_trainer(model, dataset, loss, optimizer, batch_size):
    _train_helper(model, dataset, loss, optimizer, batch_size, "cpu")


@pytest.mark.parametrize(
    "model,dataset,loss,optimizer,batch_size",
    [
        (
            TEST_MODULE,
            TestingDataset(100),
            LossWrapper(TF.mse_loss),
            SGD(TEST_MODULE.parameters(), 0.001),
            16,
        ),
        (
            TEST_MODULE,
            TestingDataset(100),
            LossWrapper(TF.mse_loss),
            Adam(TEST_MODULE.parameters()),
            27,
        ),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_module_trainer_cuda(model, dataset, loss, optimizer, batch_size):
    _train_helper(model.to("cuda"), dataset, loss, optimizer, batch_size, "cuda")


def _test_helper(model, dataset, loss, batch_size, device):
    counters = {"start": 0, "forward": 0, "loss": 0, "end": 0}
    non_counters = {"backward": 0}

    def batch_start_hook(counter, step_count, batch_size, data):
        nonlocal counters
        counters["start"] += 1

    def batch_forward_hook(counter, step_count, batch_size, data, pred):
        nonlocal counters
        counters["forward"] += 1

    def batch_loss_hook(counter, step_count, batch_size, data, pred, losses):
        nonlocal counters
        counters["loss"] += 1

    def batch_backward_hook(counter, step_count, batch_size, data, pred, losses):
        nonlocal counters
        non_counters["backward"] += 1

    def batch_end_hook(counter, step_count, batch_size, data, pred, losses):
        nonlocal counters
        counters["end"] += 1

    data_loader = DataLoader(dataset, batch_size)
    tester = ModuleTester(model, device, loss)
    hook_refs = [
        tester.run_hooks.register_batch_start_hook(batch_start_hook),
        tester.run_hooks.register_batch_forward_hook(batch_forward_hook),
        tester.run_hooks.register_batch_loss_hook(batch_loss_hook),
        tester.run_hooks.register_batch_backward_hook(batch_backward_hook),
        tester.run_hooks.register_batch_end_hook(batch_end_hook),
    ]
    result = tester.run(data_loader, desc="", track_results=False)
    assert not result

    for ref in hook_refs:
        ref.remove()

    expected_batches = math.ceil(len(dataset) / float(batch_size))
    for key, call_count in counters.items():
        assert expected_batches == call_count

    for key, call_count in non_counters.items():
        assert call_count == 0

    result = tester.run_epoch(
        data_loader, epoch=0, show_progress=False, track_results=True
    )
    assert isinstance(result, ModuleRunResults)

    for key, losses in result.results.items():
        assert len(losses) == expected_batches

        for loss in losses:
            assert not loss.is_cuda


@pytest.mark.parametrize(
    "model,dataset,loss,batch_size",
    [
        (TEST_MODULE, TestingDataset(100), LossWrapper(TF.mse_loss), 16),
        (TEST_MODULE, TestingDataset(100), LossWrapper(TF.mse_loss), 27),
    ],
)
def test_module_tester(model, dataset, loss, batch_size):
    _test_helper(model, dataset, loss, batch_size, "cpu")


@pytest.mark.parametrize(
    "model,dataset,loss,batch_size",
    [
        (TEST_MODULE, TestingDataset(100), LossWrapper(TF.mse_loss), 16),
        (TEST_MODULE, TestingDataset(100), LossWrapper(TF.mse_loss), 27),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_module_tester_cuda(model, dataset, loss, batch_size):
    _test_helper(model.to("cuda"), dataset, loss, batch_size, "cuda")
