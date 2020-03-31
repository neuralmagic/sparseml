import pytest

from neuralmagicML.pytorch.recal import (
    ScheduledModifierManager,
    ScheduledModifier,
    PyTorchModifierYAML,
)

from .test_modifier import (
    ModifierTest,
    def_model,
    def_optim_sgd,
    def_optim_adam,
    test_loss,
    test_epoch,
    test_steps_per_epoch,
)


@pytest.mark.parametrize(
    "modifier_lambda",
    [lambda: ScheduledModifierManager([ScheduledModifier()])],
    scope="function",
)
@pytest.mark.parametrize("model_lambda", [def_model], scope="function")
@pytest.mark.parametrize(
    "optim_lambda", [def_optim_sgd, def_optim_adam], scope="function"
)
class TestManagerImpl(ModifierTest):
    pass


@PyTorchModifierYAML()
class TestModifierImpl(ScheduledModifier):
    pass


def test_manager_yaml():
    manager = ScheduledModifierManager([ScheduledModifier()])
    yaml_str = ScheduledModifierManager.list_to_yaml(manager.modifiers)
    assert yaml_str
