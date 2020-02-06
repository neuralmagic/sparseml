import pytest

import torch
from torch import Tensor
from torch.nn import Module, Sequential
from torch.optim import Adam

from neuralmagicML.recal import Modifier, ScheduledModifier, ScheduledUpdateModifier


@pytest.fixture
def def_model() -> Module:
    return Sequential(*[])


@pytest.fixture
def def_optimizer() -> Adam:
    return Adam(Sequential(*[]).parameters())


@pytest.fixture
def def_loss() -> Tensor:
    return torch.tensor(0.0)


DEFAULT_EPOCH = 0.0
DEFAULT_STEPS_PER_EPOCH = 10000


@pytest.mark.parametrize('model', def_model)
@pytest.mark.parametrize('optimizer', def_optimizer)
@pytest.mark.parametrize('loss', def_loss)
def test_modifier_lifecycle(model, optimizer, loss):
    modifier = Modifier()

    with pytest.raises(NotImplementedError):
        modifier.initialize(model, optimizer)
        modifier.update(model, optimizer, DEFAULT_EPOCH, DEFAULT_STEPS_PER_EPOCH)
        modifier.loss_update(loss, model, optimizer, DEFAULT_EPOCH, DEFAULT_STEPS_PER_EPOCH)
        modifier.optimizer_pre_step(model, optimizer, DEFAULT_EPOCH, DEFAULT_STEPS_PER_EPOCH)
        modifier.optimizer_post_step(model, optimizer, DEFAULT_EPOCH, DEFAULT_STEPS_PER_EPOCH)


@pytest.mark.parametrize('model', def_model)
@pytest.mark.parametrize('optimizer', def_optimizer)
@pytest.mark.parametrize('loss', def_loss)
def test_scheduled_modifier_lifecycle(model, optimizer, loss):
    modifier = ScheduledModifier()
    modifier.initialize(model, optimizer)
    assert modifier.update_ready(DEFAULT_EPOCH, DEFAULT_STEPS_PER_EPOCH)

    with pytest.raises(NotImplementedError):
        modifier.scheduled_update(model, optimizer, DEFAULT_EPOCH, DEFAULT_STEPS_PER_EPOCH)
        modifier.update(model, optimizer, DEFAULT_EPOCH, DEFAULT_STEPS_PER_EPOCH)

    modifier.loss_update(loss, model, optimizer, DEFAULT_EPOCH, DEFAULT_STEPS_PER_EPOCH)
    modifier.optimizer_pre_step(model, optimizer, DEFAULT_EPOCH, DEFAULT_STEPS_PER_EPOCH)
    modifier.optimizer_post_step(model, optimizer, DEFAULT_EPOCH, DEFAULT_STEPS_PER_EPOCH)


@pytest.mark.parametrize('model', def_model)
@pytest.mark.parametrize('optimizer', def_optimizer)
@pytest.mark.parametrize('loss', def_loss)
def test_scheduled_modifier_start_end(model, optimizer, loss):
    prev_epoch = 0.0
    start_epoch = 1.0
    end_epoch = 10.0
    after_epoch = 15.0
    modifier = ScheduledModifier(start_epoch, end_epoch)
    modifier.initialize(model, optimizer)

    # check previous epoch
    assert not modifier.start_pending(prev_epoch, DEFAULT_STEPS_PER_EPOCH)
    assert not modifier.end_pending(prev_epoch, DEFAULT_STEPS_PER_EPOCH)
    assert not modifier.update_ready(prev_epoch, DEFAULT_STEPS_PER_EPOCH)

    # check start epoch
    assert modifier.start_pending(start_epoch, DEFAULT_STEPS_PER_EPOCH)
    assert not modifier.end_pending(start_epoch, DEFAULT_STEPS_PER_EPOCH)
    assert modifier.update_ready(start_epoch, DEFAULT_STEPS_PER_EPOCH)

    with pytest.raises(NotImplementedError):
        modifier.scheduled_update(model, optimizer, start_epoch, DEFAULT_STEPS_PER_EPOCH)

    assert modifier.started

    # check end epoch
    assert not modifier.start_pending(end_epoch, DEFAULT_STEPS_PER_EPOCH)
    assert modifier.end_pending(end_epoch, DEFAULT_STEPS_PER_EPOCH)
    assert modifier.update_ready(end_epoch, DEFAULT_STEPS_PER_EPOCH)

    with pytest.raises(NotImplementedError):
        modifier.scheduled_update(model, optimizer, end_epoch, DEFAULT_STEPS_PER_EPOCH)

    assert modifier.ended

    # check after epoch
    assert not modifier.start_pending(after_epoch, DEFAULT_STEPS_PER_EPOCH)
    assert not modifier.end_pending(after_epoch, DEFAULT_STEPS_PER_EPOCH)
    assert not modifier.update_ready(after_epoch, DEFAULT_STEPS_PER_EPOCH)


@pytest.mark.parametrize('model', def_model)
@pytest.mark.parametrize('optimizer', def_optimizer)
@pytest.mark.parametrize('loss', def_loss)
def test_scheduled_update_modifier_lifecycle(model, optimizer, loss):
    modifier = ScheduledUpdateModifier()
    modifier.initialize(model, optimizer)
    assert modifier.update_ready(DEFAULT_EPOCH, DEFAULT_STEPS_PER_EPOCH)

    with pytest.raises(NotImplementedError):
        modifier.scheduled_update(model, optimizer, DEFAULT_EPOCH, DEFAULT_STEPS_PER_EPOCH)
        modifier.update(model, optimizer, DEFAULT_EPOCH, DEFAULT_STEPS_PER_EPOCH)

    modifier.loss_update(loss, model, optimizer, DEFAULT_EPOCH, DEFAULT_STEPS_PER_EPOCH)
    modifier.optimizer_pre_step(model, optimizer, DEFAULT_EPOCH, DEFAULT_STEPS_PER_EPOCH)
    modifier.optimizer_post_step(model, optimizer, DEFAULT_EPOCH, DEFAULT_STEPS_PER_EPOCH)


@pytest.mark.parametrize('model', def_model)
@pytest.mark.parametrize('optimizer', def_optimizer)
@pytest.mark.parametrize('loss', def_loss)
def test_scheduled_update_modifier_start_end(model, optimizer, loss):
    prev_epoch = 0.0
    start_epoch = 1.0
    update_prev_epoch = 1.5
    update_start_epoch = 2.0
    update_after_epoch = 2.5
    end_epoch = 10.0
    after_epoch = 15.0
    update_frequency = 1.0
    modifier = ScheduledUpdateModifier(start_epoch, end_epoch, update_frequency)
    modifier.initialize(model, optimizer)

    # check previous epoch
    assert not modifier.start_pending(prev_epoch, DEFAULT_STEPS_PER_EPOCH)
    assert not modifier.end_pending(prev_epoch, DEFAULT_STEPS_PER_EPOCH)
    assert not modifier.update_ready(prev_epoch, DEFAULT_STEPS_PER_EPOCH)

    # check start epoch
    assert modifier.start_pending(start_epoch, DEFAULT_STEPS_PER_EPOCH)
    assert not modifier.end_pending(start_epoch, DEFAULT_STEPS_PER_EPOCH)
    assert modifier.update_ready(start_epoch, DEFAULT_STEPS_PER_EPOCH)

    with pytest.raises(NotImplementedError):
        modifier.scheduled_update(model, optimizer, start_epoch, DEFAULT_STEPS_PER_EPOCH)

    assert modifier.started

    # check end epoch
    assert not modifier.start_pending(end_epoch, DEFAULT_STEPS_PER_EPOCH)
    assert modifier.end_pending(end_epoch, DEFAULT_STEPS_PER_EPOCH)
    assert modifier.update_ready(end_epoch, DEFAULT_STEPS_PER_EPOCH)

    with pytest.raises(NotImplementedError):
        modifier.scheduled_update(model, optimizer, end_epoch, DEFAULT_STEPS_PER_EPOCH)

    assert modifier.ended

    # check after epoch
    assert not modifier.start_pending(after_epoch, DEFAULT_STEPS_PER_EPOCH)
    assert not modifier.end_pending(after_epoch, DEFAULT_STEPS_PER_EPOCH)
    assert not modifier.update_ready(after_epoch, DEFAULT_STEPS_PER_EPOCH)


@pytest.mark.parametrize('model', def_model)
@pytest.mark.parametrize('optimizer', def_optimizer)
@pytest.mark.parametrize('loss', def_loss)
def test_scheduled_update_modifier_update(model, optimizer, loss):
    start_epoch = 1.0
    end_epoch = 10.0
    update_prev_epoch = 1.5
    update_start_epoch = 2.0
    update_after_epoch = 2.5
    update_frequency = 1.0
    modifier = ScheduledUpdateModifier(start_epoch, end_epoch, update_frequency)
    modifier.initialize(model, optimizer)

    # run start epoch
    assert modifier.update_ready(start_epoch, DEFAULT_STEPS_PER_EPOCH)

    with pytest.raises(NotImplementedError):
        modifier.scheduled_update(model, optimizer, start_epoch, DEFAULT_STEPS_PER_EPOCH)

    assert modifier.started

    # check update prev epoch
    assert not modifier.update_ready(update_prev_epoch, DEFAULT_STEPS_PER_EPOCH)

    # check update epoch
    assert modifier.update_ready(update_start_epoch, DEFAULT_STEPS_PER_EPOCH)

    with pytest.raises(NotImplementedError):
        modifier.scheduled_update(model, optimizer, update_start_epoch, DEFAULT_STEPS_PER_EPOCH)

    # check update after epoch
    assert not modifier.update_ready(update_after_epoch, DEFAULT_STEPS_PER_EPOCH)

    with pytest.raises(NotImplementedError):
        modifier.scheduled_update(model, optimizer, update_after_epoch, DEFAULT_STEPS_PER_EPOCH)
