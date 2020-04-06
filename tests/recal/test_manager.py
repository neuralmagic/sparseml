from neuralmagicML.recal import BaseManager
from neuralmagicML.recal import BaseScheduled


def test_manager():
    manager = BaseManager(
        modifiers=[
            BaseScheduled(
                start_epoch=1.0,
                min_start=0,
                end_epoch=2.0,
                min_end=0,
                end_comparator=-1,
            ),
            BaseScheduled(
                start_epoch=5.0,
                min_start=0,
                end_epoch=10.0,
                min_end=0,
                end_comparator=-1,
            ),
        ]
    )
    assert manager.min_epochs == 1.0
    assert manager.max_epochs == 10.0
