import pytest

import os

from neuralmagicML.tensorflow.recal import EpochRangeModifier

from tests.tensorflow.recal.test_modifier import (
    ScheduledModifierTest,
    mlp_graph_lambda,
    conv_graph_lambda,
)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False), reason="Skipping tensorflow tests",
)
@pytest.mark.parametrize(
    "modifier_lambda",
    [lambda: EpochRangeModifier(0.0, 10.0), lambda: EpochRangeModifier(5.0, 15.0)],
    scope="function",
)
@pytest.mark.parametrize(
    "graph_lambda", [mlp_graph_lambda, conv_graph_lambda], scope="function"
)
@pytest.mark.parametrize("steps_per_epoch", [100], scope="function")
class TestEpochRangeModifierImpl(ScheduledModifierTest):
    pass


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False), reason="Skipping tensorflow tests",
)
def test_epoch_range_yaml():
    start_epoch = 5.0
    end_epoch = 15.0
    yaml_str = """
    !EpochRangeModifier
        start_epoch: {start_epoch}
        end_epoch: {end_epoch}
    """.format(
        start_epoch=start_epoch, end_epoch=end_epoch
    )
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
