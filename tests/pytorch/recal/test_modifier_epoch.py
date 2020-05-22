import pytest
import os

from neuralmagicML.pytorch.recal import EpochRangeModifier

from tests.pytorch.helpers import LinearNet, create_optim_sgd, create_optim_adam
from tests.pytorch.recal.test_modifier import (
    ScheduledModifierTest,
    test_epoch,
    test_steps_per_epoch,
    test_loss,
)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "modifier_lambda",
    [lambda: EpochRangeModifier(0.0, 10.0), lambda: EpochRangeModifier(5.0, 15.0)],
    scope="function",
)
@pytest.mark.parametrize("model_lambda", [LinearNet], scope="function")
@pytest.mark.parametrize(
    "optim_lambda", [create_optim_sgd, create_optim_adam], scope="function",
)
class TestEpochRangeModifierImpl(ScheduledModifierTest):
    pass


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
def test_epoch_range_yaml():
    start_epoch = 5.0
    end_epoch = 15.0
    yaml_str = f"""
    !EpochRangeModifier
        start_epoch: {start_epoch}
        end_epoch: {end_epoch}
    """
    yaml_modifier = EpochRangeModifier.load_obj(yaml_str)  # type: EpochRangeModifier
    serialized_modifier = EpochRangeModifier.load_obj(
        str(yaml_modifier)
    )  # type: EpochRangeModifier
    obj_modifier = EpochRangeModifier(start_epoch=start_epoch, end_epoch=end_epoch)

    assert isinstance(yaml_modifier, EpochRangeModifier)
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
