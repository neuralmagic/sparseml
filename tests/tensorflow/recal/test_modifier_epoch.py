
from neuralmagicML.tensorflow.recal import EpochRangeModifier


def test_epoch_range_modifier():
    modifier = EpochRangeModifier(start_epoch=5, end_epoch=40)
    assert modifier.start_epoch == 5
    assert modifier.end_epoch == 40

    modifier.create_ops(None, 100, None)
    modifier.create_extras(None, 100, None)
    modifier.complete_graph(None)
