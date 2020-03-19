import pytest

import sys
from torch.optim import SGD, Adam
from torch.optim.optimizer import Optimizer

from neuralmagicML.pytorch.recal import (
    SetLearningRateModifier,
    LearningRateModifier,
    CyclicLRModifier,
)

from ..test_modifier import (
    ScheduledModifierTest,
    ScheduledUpdateModifierTest,
    test_epoch,
    test_steps_per_epoch,
    test_loss,
    def_model,
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
SET_LR_MODIFIERS = [
    lambda: SetLearningRateModifier(learning_rate=SET_LR),
    lambda: SetLearningRateModifier(learning_rate=SET_LR, start_epoch=10.0),
]


@pytest.mark.parametrize("modifier_lambda", SET_LR_MODIFIERS, scope="function")
@pytest.mark.parametrize("model_lambda", [def_model], scope="function")
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
        modifier, model, optimizer = self.create_test_objs(
            modifier_lambda, model_lambda, optim_lambda
        )
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
STEP_SIZE = 5
EPOCH_APPLY_RANGE = 15
LR_STEP_MODIFIERS = [
    lambda: LearningRateModifier(
        lr_class="StepLR",
        lr_kwargs={"step_size": STEP_SIZE, "gamma": GAMMA},
        init_lr=SET_LR,
        start_epoch=0.0,
        end_epoch=EPOCH_APPLY_RANGE,
    ),
    lambda: LearningRateModifier(
        lr_class="StepLR",
        lr_kwargs={"step_size": STEP_SIZE, "gamma": GAMMA},
        init_lr=SET_LR,
        start_epoch=10.0,
        end_epoch=10 + EPOCH_APPLY_RANGE,
    ),
]


@pytest.mark.parametrize("modifier_lambda", LR_STEP_MODIFIERS, scope="function")
@pytest.mark.parametrize("model_lambda", [def_model], scope="function")
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
        modifier, model, optimizer = self.create_test_objs(
            modifier_lambda, model_lambda, optim_lambda
        )
        self.initialize_helper(modifier, model, optimizer)
        assert _get_optim_lr(optimizer) == INIT_LR

        # check optim lr is not set before
        for epoch in range(int(modifier.start_epoch)):
            assert not modifier.update_ready(epoch, test_steps_per_epoch)
            assert _get_optim_lr(optimizer) == INIT_LR

        epoch = int(modifier.start_epoch)
        assert modifier.update_ready(epoch, test_steps_per_epoch)
        modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)
        assert _get_optim_lr(optimizer) == SET_LR

        epoch_tests = []
        for index, step in enumerate(range(0, EPOCH_APPLY_RANGE, STEP_SIZE)):
            epoch_tests.extend(
                [
                    (
                        int(modifier.start_epoch) + step + counter,
                        SET_LR * GAMMA ** index,
                    )
                    for counter in range(1 if index == 0 else 0, STEP_SIZE)
                ]
            )
        epoch_tests.append((int(modifier.end_epoch), epoch_tests[-1][1] * GAMMA))

        for (epoch, test_lr) in epoch_tests:
            assert modifier.update_ready(epoch, test_steps_per_epoch)
            modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)
            assert abs(_get_optim_lr(optimizer) - test_lr) < sys.float_info.epsilon

        for epoch in range(int(modifier.end_epoch) + 1, int(modifier.end_epoch) + 10):
            assert not modifier.update_ready(epoch, test_steps_per_epoch)
            assert (
                abs(_get_optim_lr(optimizer) - epoch_tests[-1][1])
                < sys.float_info.epsilon
            )


def test_lr_modifier_step_yaml():
    lr_class = "StepLR"
    lr_kwargs = {"step_size": STEP_SIZE, "gamma": GAMMA}
    start_epoch = 10.0
    end_epoch = 20.0
    update_frequency = 1.0
    init_lr = 0.1
    yaml_str = """
    !LearningRateModifier
        start_epoch: {}
        end_epoch: {}
        update_frequency: {}
        lr_class: {}
        lr_kwargs: {}
        init_lr: {}
    """.format(
        start_epoch, end_epoch, update_frequency, lr_class, lr_kwargs, init_lr
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
        update_frequency=update_frequency,
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
LR_MULTI_STEP_MODIFIERS = [
    lambda: LearningRateModifier(
        lr_class="MultiStepLR",
        lr_kwargs={"milestones": MILESTONES, "gamma": GAMMA},
        init_lr=SET_LR,
        start_epoch=0.0,
        end_epoch=MILESTONES[-1] + 5,
    ),
    lambda: LearningRateModifier(
        lr_class="MultiStepLR",
        lr_kwargs={"milestones": [m + 10 for m in MILESTONES], "gamma": GAMMA},
        init_lr=SET_LR,
        start_epoch=10.0,
        end_epoch=10 + MILESTONES[-1] + 5,
    ),
]


@pytest.mark.parametrize("modifier_lambda", LR_MULTI_STEP_MODIFIERS, scope="function")
@pytest.mark.parametrize("model_lambda", [def_model], scope="function")
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
        modifier, model, optimizer = self.create_test_objs(
            modifier_lambda, model_lambda, optim_lambda
        )
        self.initialize_helper(modifier, model, optimizer)
        assert _get_optim_lr(optimizer) == INIT_LR

        # check optim lr is not set before
        for epoch in range(int(modifier.start_epoch)):
            assert not modifier.update_ready(epoch, test_steps_per_epoch)
            assert _get_optim_lr(optimizer) == INIT_LR

        epoch = int(modifier.start_epoch)
        assert modifier.update_ready(epoch, test_steps_per_epoch)
        modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)
        assert _get_optim_lr(optimizer) == SET_LR

        epoch_tests = []
        lr = SET_LR
        for counter in range(
            int(modifier.start_epoch) + 1, int(modifier.end_epoch) + 1
        ):
            if counter - int(modifier.start_epoch) in MILESTONES:
                lr = lr * GAMMA

            epoch_tests.append((counter, lr))

        for (epoch, test_lr) in epoch_tests:
            assert modifier.update_ready(epoch, test_steps_per_epoch)
            modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)
            assert abs(_get_optim_lr(optimizer) - test_lr) < sys.float_info.epsilon

        for epoch in range(int(modifier.end_epoch) + 1, int(modifier.end_epoch) + 10):
            assert not modifier.update_ready(epoch, test_steps_per_epoch)
            assert (
                abs(_get_optim_lr(optimizer) - epoch_tests[-1][1])
                < sys.float_info.epsilon
            )


def test_lr_modifier_multi_step_yaml():
    lr_class = "MultiStepLR"
    lr_kwargs = {"milestones": MILESTONES, "gamma": GAMMA}
    start_epoch = 10.0
    end_epoch = 20.0
    update_frequency = 1.0
    init_lr = 0.1
    yaml_str = """
    !LearningRateModifier
        start_epoch: {}
        end_epoch: {}
        update_frequency: {}
        lr_class: {}
        lr_kwargs: {}
        init_lr: {}
    """.format(
        start_epoch, end_epoch, update_frequency, lr_class, lr_kwargs, init_lr
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
        update_frequency=update_frequency,
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


LR_EXPONENTIAL_MODIFIERS = [
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
]


@pytest.mark.parametrize("modifier_lambda", LR_EXPONENTIAL_MODIFIERS, scope="function")
@pytest.mark.parametrize("model_lambda", [def_model], scope="function")
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
        modifier, model, optimizer = self.create_test_objs(
            modifier_lambda, model_lambda, optim_lambda
        )
        self.initialize_helper(modifier, model, optimizer)
        assert _get_optim_lr(optimizer) == INIT_LR

        # check optim lr is not set before
        for epoch in range(int(modifier.start_epoch)):
            assert not modifier.update_ready(epoch, test_steps_per_epoch)
            assert _get_optim_lr(optimizer) == INIT_LR

        epoch = int(modifier.start_epoch)
        assert modifier.update_ready(epoch, test_steps_per_epoch)
        modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)
        assert _get_optim_lr(optimizer) == SET_LR

        lr = SET_LR
        for epoch in range(int(modifier.start_epoch) + 1, int(modifier.end_epoch) + 1):
            lr = lr * GAMMA
            assert modifier.update_ready(epoch, test_steps_per_epoch)
            modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)
            assert abs(_get_optim_lr(optimizer) - lr) < sys.float_info.epsilon

        for epoch in range(int(modifier.end_epoch) + 1, int(modifier.end_epoch) + 10):
            assert not modifier.update_ready(epoch, test_steps_per_epoch)
            assert abs(_get_optim_lr(optimizer) - lr) < sys.float_info.epsilon


def test_lr_modifier_exponential_yaml():
    lr_class = "ExponentialLR"
    lr_kwargs = {"gamma": GAMMA}
    start_epoch = 10.0
    end_epoch = 20.0
    update_frequency = 1.0
    init_lr = 0.1
    yaml_str = """
    !LearningRateModifier
        start_epoch: {}
        end_epoch: {}
        update_frequency: {}
        lr_class: {}
        lr_kwargs: {}
        init_lr: {}
    """.format(
        start_epoch, end_epoch, update_frequency, lr_class, lr_kwargs, init_lr
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
        update_frequency=update_frequency,
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


##############################
#
# CyclicLRModifier functions
#
##############################
CYCLIC_START = 0.001
CYCLIC_END = 0.1
LR_CYCLIC_MODIFIERS = [
    lambda: CyclicLRModifier(
        lr_kwargs={
            "base_lr": CYCLIC_START,
            "max_lr": CYCLIC_END,
            "step_size_up": 100,
            "step_size_down": 100,
        },
        start_epoch=0.0,
        end_epoch=15.0,
    ),
    lambda: CyclicLRModifier(
        lr_kwargs={
            "base_lr": CYCLIC_START,
            "max_lr": CYCLIC_END,
            "step_size_up": 100,
            "step_size_down": 100,
        },
        start_epoch=10.0,
        end_epoch=25.0,
    ),
]


@pytest.mark.parametrize("modifier_lambda", LR_CYCLIC_MODIFIERS, scope="function")
@pytest.mark.parametrize("model_lambda", [def_model], scope="function")
@pytest.mark.parametrize(
    "optim_lambda", [lambda model: SGD(model.parameters(), INIT_LR)], scope="function",
)
class TestLRModifierCyclicImpl(ScheduledUpdateModifierTest):
    def test_lifecycle(
        self, modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
    ):
        modifier, model, optimizer = self.create_test_objs(
            modifier_lambda, model_lambda, optim_lambda
        )
        self.initialize_helper(modifier, model, optimizer)
        assert _get_optim_lr(optimizer) == INIT_LR

        # check optim lr is not set before
        for epoch in range(int(modifier.start_epoch)):
            assert not modifier.update_ready(epoch, test_steps_per_epoch)
            assert _get_optim_lr(optimizer) == INIT_LR

        epoch = int(modifier.start_epoch)
        assert modifier.update_ready(epoch, test_steps_per_epoch)
        modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)
        assert _get_optim_lr(optimizer) == CYCLIC_START

        increase = True
        for epoch in range(int(modifier.start_epoch) + 1, int(modifier.end_epoch)):
            increase = not increase
            check_lr = CYCLIC_START - 0.000001 if increase else CYCLIC_END + 0.000001

            for batch in range(test_steps_per_epoch):
                epoch_batch = epoch + float(batch) / float(test_steps_per_epoch)
                assert modifier.update_ready(epoch_batch, test_steps_per_epoch)
                modifier.scheduled_update(
                    model, optimizer, epoch_batch, test_steps_per_epoch
                )
                optim_lr = _get_optim_lr(optimizer)

                if increase:
                    assert optim_lr > check_lr
                else:
                    assert optim_lr < check_lr

                check_lr = optim_lr

        epoch = int(modifier.end_epoch)
        assert modifier.update_ready(epoch, test_steps_per_epoch)
        modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)

        for epoch in range(int(modifier.end_epoch) + 1, int(modifier.end_epoch) + 10):
            assert not modifier.update_ready(epoch, test_steps_per_epoch)


def test_lr_modifier_cyclic_yaml():
    lr_kwargs = {
        "base_lr": CYCLIC_START,
        "max_lr": CYCLIC_END,
        "step_size_up": 100,
        "step_size_down": 100,
    }
    start_epoch = 10.0
    end_epoch = 20.0
    yaml_str = """
    !CyclicLRModifier
        start_epoch: {}
        end_epoch: {}
        lr_kwargs: {}
    """.format(
        start_epoch, end_epoch, lr_kwargs
    )
    yaml_modifier = CyclicLRModifier.load_obj(yaml_str)  # type: CyclicLRModifier
    serialized_modifier = CyclicLRModifier.load_obj(
        str(yaml_modifier)
    )  # type: CyclicLRModifier
    obj_modifier = CyclicLRModifier(
        start_epoch=start_epoch, end_epoch=end_epoch, lr_kwargs=lr_kwargs
    )

    assert isinstance(yaml_modifier, CyclicLRModifier)
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
        yaml_modifier.lr_kwargs
        == serialized_modifier.lr_kwargs
        == obj_modifier.lr_kwargs
    )
