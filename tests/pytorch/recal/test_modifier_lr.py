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
        lambda: SetLearningRateModifier(learning_rate=SET_LR),
        lambda: SetLearningRateModifier(learning_rate=SET_LR, start_epoch=10.0),
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

        # check optim lr is not set before
        for epoch in range(int(modifier.start_epoch)):
            assert not modifier.update_ready(epoch, test_steps_per_epoch)
            assert modifier.applied_learning_rate < 0
            assert _get_optim_lr(optimizer) == INIT_LR

        epoch = int(modifier.start_epoch)
        assert modifier.update_ready(epoch, test_steps_per_epoch)
        modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)
        assert modifier.applied_learning_rate == SET_LR
        assert _get_optim_lr(optimizer) == SET_LR

        # check optim lr is set after start
        for epoch in range(
            int(modifier.start_epoch) + 1, int(modifier.start_epoch) + 10
        ):
            assert not modifier.update_ready(epoch, test_steps_per_epoch)
            assert modifier.applied_learning_rate == SET_LR
            assert _get_optim_lr(optimizer) == SET_LR


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
# LearningRateModifier functions
#
##############################

GAMMA = 0.1
EPOCH_APPLY_RANGE = 15


@pytest.mark.parametrize(
    "modifier_lambda",
    [
        lambda: LearningRateModifier(
            lr_class="StepLR",
            lr_kwargs={"gamma": GAMMA, "step_size": 1.0},
            init_lr=SET_LR,
            start_epoch=0.0,
            end_epoch=EPOCH_APPLY_RANGE,
        ),
        lambda: LearningRateModifier(
            lr_class="StepLR",
            lr_kwargs={"gamma": GAMMA, "step_size": 2.0},
            init_lr=SET_LR,
            start_epoch=5.0,
            end_epoch=5 + EPOCH_APPLY_RANGE,
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
            update_perc = (epoch - modifier.start_epoch) / (
                modifier.lr_kwargs["step_size"] / test_steps_per_epoch
            )
            num_gammas = math.floor(update_perc)
            expected_lr = (
                SET_LR * modifier.lr_kwargs["gamma"] ** num_gammas
                if num_gammas > 0
                else SET_LR
            )

            for step in range(test_steps_per_epoch):
                test_epoch = float(epoch) + float(step) / float(test_steps_per_epoch)

                if test_epoch < modifier.start_epoch:
                    assert not modifier.update_ready(test_epoch, test_steps_per_epoch)
                    optim_lr = _get_optim_lr(optimizer)
                    assert optim_lr == INIT_LR
                elif abs(test_epoch - modifier.start_epoch) < sys.float_info.epsilon:
                    assert modifier.update_ready(test_epoch, test_steps_per_epoch)
                    modifier.scheduled_update(
                        model, optimizer, test_epoch, test_steps_per_epoch
                    )
                    optim_lr = _get_optim_lr(optimizer)
                    assert optim_lr == SET_LR
                elif test_epoch < modifier.end_epoch:
                    assert modifier.update_ready(test_epoch, test_steps_per_epoch)
                    modifier.scheduled_update(
                        model, optimizer, test_epoch, test_steps_per_epoch
                    )
                    optim_lr = _get_optim_lr(optimizer)
                    assert abs(optim_lr - expected_lr) < 1e-5
                elif abs(test_epoch - modifier.end_epoch) < sys.float_info.epsilon:
                    assert modifier.update_ready(test_epoch, test_steps_per_epoch)
                    modifier.scheduled_update(
                        model, optimizer, test_epoch, test_steps_per_epoch
                    )
                    optim_lr = _get_optim_lr(optimizer)
                    assert abs(optim_lr - expected_lr) < 1e-5
                else:
                    assert not modifier.update_ready(test_epoch, test_steps_per_epoch)
                    optim_lr = _get_optim_lr(optimizer)
                    assert abs(optim_lr - expected_lr) < 1e-5


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
            lr_kwargs={"milestones": MILESTONES, "gamma": GAMMA},
            init_lr=SET_LR,
            start_epoch=0.0,
            end_epoch=MILESTONES[-1] + 5,
        ),
        lambda: LearningRateModifier(
            lr_class="MultiStepLR",
            lr_kwargs={"milestones": [m + 2 for m in MILESTONES], "gamma": GAMMA},
            init_lr=SET_LR,
            start_epoch=2.0,
            end_epoch=MILESTONES[-1] + 5,
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
            num_gammas = (
                0
                if not modifier._lr_scheduler
                else sum(
                    [
                        1
                        if epoch - modifier.start_epoch
                        >= round(mile / float(test_steps_per_epoch))
                        else 0
                        for mile in modifier.lr_kwargs["milestones"]
                    ]
                )
            )
            expected_lr = (
                SET_LR * modifier.lr_kwargs["gamma"] ** num_gammas
                if num_gammas > 0
                else SET_LR
            )

            for step in range(test_steps_per_epoch):
                test_epoch = float(epoch) + float(step) / float(test_steps_per_epoch)

                if test_epoch < modifier.start_epoch:
                    assert not modifier.update_ready(test_epoch, test_steps_per_epoch)
                    optim_lr = _get_optim_lr(optimizer)
                    assert optim_lr == INIT_LR
                elif abs(test_epoch - modifier.start_epoch) < sys.float_info.epsilon:
                    assert modifier.update_ready(test_epoch, test_steps_per_epoch)
                    modifier.scheduled_update(
                        model, optimizer, test_epoch, test_steps_per_epoch
                    )
                    optim_lr = _get_optim_lr(optimizer)
                    assert optim_lr == SET_LR
                elif test_epoch < modifier.end_epoch:
                    assert modifier.update_ready(test_epoch, test_steps_per_epoch)
                    modifier.scheduled_update(
                        model, optimizer, test_epoch, test_steps_per_epoch
                    )
                    optim_lr = _get_optim_lr(optimizer)
                    assert abs(optim_lr - expected_lr) < 1e-5
                elif abs(test_epoch - modifier.end_epoch) < sys.float_info.epsilon:
                    assert modifier.update_ready(test_epoch, test_steps_per_epoch)
                    modifier.scheduled_update(
                        model, optimizer, test_epoch, test_steps_per_epoch
                    )
                    optim_lr = _get_optim_lr(optimizer)
                    assert abs(optim_lr - expected_lr) < 1e-5
                else:
                    assert not modifier.update_ready(test_epoch, test_steps_per_epoch)
                    optim_lr = _get_optim_lr(optimizer)
                    assert abs(optim_lr - expected_lr) < 1e-5


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


@pytest.mark.parametrize(
    "modifier_lambda",
    [
        lambda: LearningRateModifier(
            lr_class="ExponentialLR",
            lr_kwargs={"gamma": GAMMA},
            init_lr=SET_LR,
            start_epoch=0.0,
            end_epoch=15.0,
        ),
        lambda: LearningRateModifier(
            lr_class="ExponentialLR",
            lr_kwargs={"gamma": GAMMA},
            init_lr=SET_LR,
            start_epoch=10.0,
            end_epoch=25.0,
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
            num_gammas = math.floor(epoch - modifier.start_epoch)
            expected_lr = (
                SET_LR * modifier.lr_kwargs["gamma"] ** num_gammas
                if num_gammas > 0
                else SET_LR
            )

            for step in range(test_steps_per_epoch):
                test_epoch = float(epoch) + float(step) / float(test_steps_per_epoch)

                if test_epoch < modifier.start_epoch:
                    assert not modifier.update_ready(test_epoch, test_steps_per_epoch)
                    optim_lr = _get_optim_lr(optimizer)
                    assert optim_lr == INIT_LR
                elif abs(test_epoch - modifier.start_epoch) < sys.float_info.epsilon:
                    assert modifier.update_ready(test_epoch, test_steps_per_epoch)
                    modifier.scheduled_update(
                        model, optimizer, test_epoch, test_steps_per_epoch
                    )
                    optim_lr = _get_optim_lr(optimizer)
                    assert optim_lr == SET_LR
                elif test_epoch < modifier.end_epoch:
                    assert modifier.update_ready(test_epoch, test_steps_per_epoch)
                    modifier.scheduled_update(
                        model, optimizer, test_epoch, test_steps_per_epoch
                    )
                    optim_lr = _get_optim_lr(optimizer)
                    assert abs(optim_lr - expected_lr) < 1e-5
                elif abs(test_epoch - modifier.end_epoch) < sys.float_info.epsilon:
                    assert modifier.update_ready(test_epoch, test_steps_per_epoch)
                    modifier.scheduled_update(
                        model, optimizer, test_epoch, test_steps_per_epoch
                    )
                    optim_lr = _get_optim_lr(optimizer)
                    assert abs(optim_lr - expected_lr) < 1e-5
                else:
                    assert not modifier.update_ready(test_epoch, test_steps_per_epoch)
                    optim_lr = _get_optim_lr(optimizer)
                    assert abs(optim_lr - expected_lr) < 1e-5


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
