import pytest

import sys
import math
from torch.optim import SGD, Adam
from torch.optim.optimizer import Optimizer

from neuralmagicML.pytorch.recal import (
    SetLearningRateModifier,
    LearningRateModifier,
)

from tests.pytorch.helpers import LinearNet
from tests.pytorch.recal.test_modifier import (
    ScheduledModifierTest,
    ScheduledUpdateModifierTest,
    test_epoch,
    test_steps_per_epoch,
    test_loss,
)


EPSILON = 1e-7
INIT_LR = 0.0001
SET_LR = 0.1


def _get_optim_lr(optim: Optimizer) -> float:
    for param_group in optim.param_groups:
        return param_group["lr"]


##############################
#
# SetLearningRateModifier tests
#
##############################


@pytest.mark.parametrize(
    "modifier_lambda",
    [
        lambda: SetLearningRateModifier(learning_rate=0.1),
        lambda: SetLearningRateModifier(learning_rate=0.03, start_epoch=5),
    ],
    scope="function",
)
@pytest.mark.parametrize("model_lambda", [LinearNet], scope="function")
@pytest.mark.parametrize(
    "optim_lambda",
    [
        lambda model: SGD(model.parameters(), INIT_LR),
        lambda model: Adam(model.parameters(), INIT_LR),
    ],
    scope="function",
)
class TestSetLRModifierImpl(ScheduledModifierTest):
    def test_lifecycle(
        self, modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
    ):
        modifier = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)
        self.initialize_helper(modifier, model, optimizer)
        assert modifier.applied_learning_rate < 0
        assert _get_optim_lr(optimizer) == INIT_LR

        for epoch in range(int(modifier.start_epoch) + 10):
            expected = (
                INIT_LR if epoch < modifier.start_epoch else modifier.learning_rate
            )

            for step in range(test_steps_per_epoch):
                test_epoch = float(epoch) + float(step) / float(test_steps_per_epoch)

                if epoch < modifier.start_epoch:
                    assert not modifier.update_ready(test_epoch, test_steps_per_epoch)
                elif (
                    epoch == modifier.start_epoch
                    or (modifier.start_epoch == -1 and epoch == 0)
                ) and step == 0:
                    assert modifier.update_ready(test_epoch, test_steps_per_epoch)
                    modifier.scheduled_update(
                        model, optimizer, test_epoch, test_steps_per_epoch
                    )
                else:
                    assert not modifier.update_ready(test_epoch, test_steps_per_epoch)

                if epoch >= modifier.start_epoch:
                    assert (
                        abs(modifier.applied_learning_rate - expected) < EPSILON
                    ), "Failed at epoch:{} step:{}".format(epoch, step)
                else:
                    assert (
                        abs(modifier.applied_learning_rate - -1.0) < EPSILON
                    ), "Failed at epoch:{} step:{}".format(epoch, step)

                assert (
                    abs(_get_optim_lr(optimizer) - expected) < EPSILON
                ), "Failed at epoch:{} step:{}".format(epoch, step)


def test_set_lr_yaml():
    start_epoch = 10.0
    yaml_str = """
    !SetLearningRateModifier
        learning_rate: {}
        start_epoch: {}
    """.format(
        SET_LR, start_epoch
    )
    yaml_modifier = SetLearningRateModifier.load_obj(
        yaml_str
    )  # type: SetLearningRateModifier
    serialized_modifier = SetLearningRateModifier.load_obj(
        str(yaml_modifier)
    )  # type: SetLearningRateModifier
    obj_modifier = SetLearningRateModifier(
        learning_rate=SET_LR, start_epoch=start_epoch
    )

    assert isinstance(yaml_modifier, SetLearningRateModifier)
    assert (
        yaml_modifier.learning_rate
        == serialized_modifier.learning_rate
        == obj_modifier.learning_rate
    )
    assert (
        yaml_modifier.start_epoch
        == serialized_modifier.start_epoch
        == obj_modifier.start_epoch
    )


##############################
#
# LearningRateModifier tests
#
##############################

GAMMA = 0.1
EPOCH_APPLY_RANGE = 15


@pytest.mark.parametrize(
    "modifier_lambda",
    [
        lambda: LearningRateModifier(
            lr_class="ExponentialLR",
            lr_kwargs={"gamma": 0.9},
            start_epoch=0,
            init_lr=0.1,
        ),
        lambda: LearningRateModifier(
            lr_class="ExponentialLR",
            lr_kwargs={"gamma": 0.5},
            start_epoch=5,
            end_epoch=13,
            init_lr=0.1,
        ),
    ],
    scope="function",
)
@pytest.mark.parametrize("model_lambda", [LinearNet], scope="function")
@pytest.mark.parametrize(
    "optim_lambda",
    [
        lambda model: SGD(model.parameters(), INIT_LR),
        lambda model: Adam(model.parameters(), INIT_LR),
    ],
    scope="function",
)
class TestLRModifierExponentialImpl(ScheduledUpdateModifierTest):
    def test_lifecycle(
        self, modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
    ):
        modifier = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)
        self.initialize_helper(modifier, model, optimizer)
        assert _get_optim_lr(optimizer) == INIT_LR

        for epoch in range(int(modifier.end_epoch) + 5):
            if epoch < modifier.start_epoch:
                expected = INIT_LR
            elif epoch < modifier.end_epoch or modifier.end_epoch == -1:
                expected = modifier.init_lr * (
                    modifier.lr_kwargs["gamma"] ** (epoch - int(modifier.start_epoch))
                )
            else:
                expected = modifier.init_lr * (
                    modifier.lr_kwargs["gamma"]
                    ** int(modifier.end_epoch - modifier.start_epoch - 1)
                )

            for step in range(test_steps_per_epoch):
                test_epoch = float(epoch) + float(step) / float(test_steps_per_epoch)

                if test_epoch < modifier.start_epoch:
                    assert not modifier.update_ready(test_epoch, test_steps_per_epoch)
                elif abs(test_epoch - modifier.start_epoch) < sys.float_info.epsilon:
                    assert modifier.update_ready(test_epoch, test_steps_per_epoch)
                    modifier.scheduled_update(
                        model, optimizer, test_epoch, test_steps_per_epoch
                    )
                elif test_epoch < modifier.end_epoch or modifier.end_epoch == -1:
                    assert modifier.update_ready(test_epoch, test_steps_per_epoch)
                    modifier.scheduled_update(
                        model, optimizer, test_epoch, test_steps_per_epoch
                    )
                elif abs(test_epoch - modifier.end_epoch) < sys.float_info.epsilon:
                    assert modifier.update_ready(test_epoch, test_steps_per_epoch)
                    modifier.scheduled_update(
                        model, optimizer, test_epoch, test_steps_per_epoch
                    )
                else:
                    assert not modifier.update_ready(test_epoch, test_steps_per_epoch)

                assert (
                    abs(_get_optim_lr(optimizer) - expected) < EPSILON
                ), "Failed at epoch:{} step:{}".format(epoch, step)


def test_lr_modifier_exponential_yaml():
    lr_class = "ExponentialLR"
    lr_kwargs = {"gamma": GAMMA}
    start_epoch = 10.0
    end_epoch = 20.0
    init_lr = 0.1
    yaml_str = """
    !LearningRateModifier
        start_epoch: {}
        end_epoch: {}
        lr_class: {}
        lr_kwargs: {}
        init_lr: {}
    """.format(
        start_epoch, end_epoch, lr_class, lr_kwargs, init_lr
    )
    yaml_modifier = LearningRateModifier.load_obj(
        yaml_str
    )  # type: LearningRateModifier
    serialized_modifier = LearningRateModifier.load_obj(
        str(yaml_modifier)
    )  # type: LearningRateModifier
    obj_modifier = LearningRateModifier(
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        lr_class=lr_class,
        lr_kwargs=lr_kwargs,
        init_lr=init_lr,
    )

    assert isinstance(yaml_modifier, LearningRateModifier)
    assert (
        yaml_modifier.start_epoch
        == serialized_modifier.start_epoch
        == obj_modifier.start_epoch
    )
    assert (
        yaml_modifier.end_epoch
        == serialized_modifier.end_epoch
        == obj_modifier.end_epoch
    )
    assert (
        yaml_modifier.update_frequency
        == serialized_modifier.update_frequency
        == obj_modifier.update_frequency
    )
    assert (
        yaml_modifier.lr_class == serialized_modifier.lr_class == obj_modifier.lr_class
    )
    assert (
        yaml_modifier.lr_kwargs
        == serialized_modifier.lr_kwargs
        == obj_modifier.lr_kwargs
    )
    assert yaml_modifier.init_lr == serialized_modifier.init_lr == obj_modifier.init_lr


@pytest.mark.parametrize(
    "modifier_lambda",
    [
        lambda: LearningRateModifier(
            lr_class="StepLR",
            lr_kwargs={"gamma": 0.9, "step_size": 3},
            start_epoch=0,
            init_lr=0.1,
        ),
        lambda: LearningRateModifier(
            lr_class="StepLR",
            lr_kwargs={"gamma": 0.5, "step_size": 2},
            start_epoch=5,
            end_epoch=11,
            init_lr=0.01,
        ),
    ],
    scope="function",
)
@pytest.mark.parametrize("model_lambda", [LinearNet], scope="function")
@pytest.mark.parametrize(
    "optim_lambda",
    [
        lambda model: SGD(model.parameters(), INIT_LR),
        lambda model: Adam(model.parameters(), INIT_LR),
    ],
    scope="function",
)
class TestLRModifierStepImpl(ScheduledUpdateModifierTest):
    def test_lifecycle(
        self, modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
    ):
        modifier = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)
        self.initialize_helper(modifier, model, optimizer)
        assert _get_optim_lr(optimizer) == INIT_LR

        for epoch in range(int(modifier.end_epoch) + 5):
            if epoch < modifier.start_epoch:
                expected = INIT_LR
            elif epoch < modifier.end_epoch or modifier.end_epoch == -1:
                expected = modifier.init_lr * (
                    modifier.lr_kwargs["gamma"]
                    ** math.floor(
                        (epoch - modifier.start_epoch) / modifier.lr_kwargs["step_size"]
                    )
                )
            else:
                expected = modifier.init_lr * (
                    modifier.lr_kwargs["gamma"]
                    ** (
                        math.floor(
                            (modifier.end_epoch - modifier.start_epoch)
                            / modifier.lr_kwargs["step_size"]
                        )
                        - 1
                    )
                )

            for step in range(test_steps_per_epoch):
                test_epoch = float(epoch) + float(step) / float(test_steps_per_epoch)

                if test_epoch < modifier.start_epoch:
                    assert not modifier.update_ready(test_epoch, test_steps_per_epoch)
                elif abs(test_epoch - modifier.start_epoch) < sys.float_info.epsilon:
                    assert modifier.update_ready(test_epoch, test_steps_per_epoch)
                    modifier.scheduled_update(
                        model, optimizer, test_epoch, test_steps_per_epoch
                    )
                elif test_epoch < modifier.end_epoch or modifier.end_epoch == -1:
                    assert modifier.update_ready(test_epoch, test_steps_per_epoch)
                    modifier.scheduled_update(
                        model, optimizer, test_epoch, test_steps_per_epoch
                    )
                elif abs(test_epoch - modifier.end_epoch) < sys.float_info.epsilon:
                    assert modifier.update_ready(test_epoch, test_steps_per_epoch)
                    modifier.scheduled_update(
                        model, optimizer, test_epoch, test_steps_per_epoch
                    )
                else:
                    assert not modifier.update_ready(test_epoch, test_steps_per_epoch)

                assert (
                    abs(_get_optim_lr(optimizer) - expected) < EPSILON
                ), "Failed at epoch:{} step:{}".format(epoch, step)


def test_lr_modifier_step_yaml():
    lr_class = "StepLR"
    lr_kwargs = {"step_size": 1.0, "gamma": GAMMA}
    start_epoch = 10.0
    end_epoch = 20.0
    init_lr = 0.1
    yaml_str = """
    !LearningRateModifier
        start_epoch: {}
        end_epoch: {}
        lr_class: {}
        lr_kwargs: {}
        init_lr: {}
    """.format(
        start_epoch, end_epoch, lr_class, lr_kwargs, init_lr
    )
    yaml_modifier = LearningRateModifier.load_obj(
        yaml_str
    )  # type: LearningRateModifier
    serialized_modifier = LearningRateModifier.load_obj(
        str(yaml_modifier)
    )  # type: LearningRateModifier
    obj_modifier = LearningRateModifier(
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        lr_class=lr_class,
        lr_kwargs=lr_kwargs,
        init_lr=init_lr,
    )

    assert isinstance(yaml_modifier, LearningRateModifier)
    assert (
        yaml_modifier.start_epoch
        == serialized_modifier.start_epoch
        == obj_modifier.start_epoch
    )
    assert (
        yaml_modifier.end_epoch
        == serialized_modifier.end_epoch
        == obj_modifier.end_epoch
    )
    assert (
        yaml_modifier.update_frequency
        == serialized_modifier.update_frequency
        == obj_modifier.update_frequency
    )
    assert (
        yaml_modifier.lr_class == serialized_modifier.lr_class == obj_modifier.lr_class
    )
    assert (
        yaml_modifier.lr_kwargs
        == serialized_modifier.lr_kwargs
        == obj_modifier.lr_kwargs
    )
    assert yaml_modifier.init_lr == serialized_modifier.init_lr == obj_modifier.init_lr


MILESTONES = [5, 9, 12]


@pytest.mark.parametrize(
    "modifier_lambda",
    [
        lambda: LearningRateModifier(
            lr_class="MultiStepLR",
            lr_kwargs={"gamma": 0.9, "milestones": [1, 3, 4]},
            start_epoch=0,
            init_lr=0.1,
        ),
        lambda: LearningRateModifier(
            lr_class="MultiStepLR",
            lr_kwargs={"gamma": 0.95, "milestones": [5, 8]},
            start_epoch=3,
            end_epoch=13,
            init_lr=0.1,
        ),
    ],
    scope="function",
)
@pytest.mark.parametrize("model_lambda", [LinearNet], scope="function")
@pytest.mark.parametrize(
    "optim_lambda",
    [
        lambda model: SGD(model.parameters(), INIT_LR),
        lambda model: Adam(model.parameters(), INIT_LR),
    ],
    scope="function",
)
class TestLRModifierMultiStepImpl(ScheduledUpdateModifierTest):
    def test_lifecycle(
        self, modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
    ):
        modifier = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)
        self.initialize_helper(modifier, model, optimizer)
        assert _get_optim_lr(optimizer) == INIT_LR

        for epoch in range(int(modifier.end_epoch) + 5):
            if epoch < modifier.start_epoch:
                expected = INIT_LR
            else:
                num_gammas = sum(
                    [1 for mile in modifier.lr_kwargs["milestones"] if epoch >= mile]
                )
                expected = modifier.init_lr * modifier.lr_kwargs["gamma"] ** num_gammas

            for step in range(test_steps_per_epoch):
                test_epoch = float(epoch) + float(step) / float(test_steps_per_epoch)

                if test_epoch < modifier.start_epoch:
                    assert not modifier.update_ready(test_epoch, test_steps_per_epoch)
                elif abs(test_epoch - modifier.start_epoch) < sys.float_info.epsilon:
                    assert modifier.update_ready(test_epoch, test_steps_per_epoch)
                    modifier.scheduled_update(
                        model, optimizer, test_epoch, test_steps_per_epoch
                    )
                elif test_epoch < modifier.end_epoch or modifier.end_epoch == -1:
                    assert modifier.update_ready(test_epoch, test_steps_per_epoch)
                    modifier.scheduled_update(
                        model, optimizer, test_epoch, test_steps_per_epoch
                    )
                elif abs(test_epoch - modifier.end_epoch) < sys.float_info.epsilon:
                    assert modifier.update_ready(test_epoch, test_steps_per_epoch)
                    modifier.scheduled_update(
                        model, optimizer, test_epoch, test_steps_per_epoch
                    )
                else:
                    assert not modifier.update_ready(test_epoch, test_steps_per_epoch)

                optim_lr = _get_optim_lr(optimizer)
                assert (
                    abs(optim_lr - expected) < EPSILON
                ), "Failed at epoch:{} step:{}".format(epoch, step)


def test_lr_modifier_multi_step_yaml():
    lr_class = "MultiStepLR"
    lr_kwargs = {"milestones": MILESTONES, "gamma": GAMMA}
    start_epoch = 2.0
    end_epoch = 20.0
    init_lr = 0.1
    yaml_str = """
    !LearningRateModifier
        start_epoch: {}
        end_epoch: {}
        lr_class: {}
        lr_kwargs: {}
        init_lr: {}
    """.format(
        start_epoch, end_epoch, lr_class, lr_kwargs, init_lr
    )
    yaml_modifier = LearningRateModifier.load_obj(
        yaml_str
    )  # type: LearningRateModifier
    serialized_modifier = LearningRateModifier.load_obj(
        str(yaml_modifier)
    )  # type: LearningRateModifier
    obj_modifier = LearningRateModifier(
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        lr_class=lr_class,
        lr_kwargs=lr_kwargs,
        init_lr=init_lr,
    )

    assert isinstance(yaml_modifier, LearningRateModifier)
    assert (
        yaml_modifier.start_epoch
        == serialized_modifier.start_epoch
        == obj_modifier.start_epoch
    )
    assert (
        yaml_modifier.end_epoch
        == serialized_modifier.end_epoch
        == obj_modifier.end_epoch
    )
    assert (
        yaml_modifier.update_frequency
        == serialized_modifier.update_frequency
        == obj_modifier.update_frequency
    )
    assert (
        yaml_modifier.lr_class == serialized_modifier.lr_class == obj_modifier.lr_class
    )
    assert (
        yaml_modifier.lr_kwargs
        == serialized_modifier.lr_kwargs
        == obj_modifier.lr_kwargs
    )
    assert yaml_modifier.init_lr == serialized_modifier.init_lr == obj_modifier.init_lr
