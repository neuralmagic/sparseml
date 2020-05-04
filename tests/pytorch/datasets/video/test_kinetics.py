from torch.utils.data import Dataset

from neuralmagicML.pytorch.datasets import (
    DatasetRegistry,
    Kinetics400Dataset,
)


def _validate_kinetics(dataset: Dataset, size: int):
    assert len(dataset) > 0
    item = dataset[0]
    assert isinstance(item, tuple)
    assert item[0].shape[0] == 3
    assert item[0].shape[1] == 30
    assert item[0].shape[2] == size
    assert item[0].shape[3] == size
    assert len(item[1]) > 0
    assert isinstance(item[2], int)


def test_kinetics_detection():
    train_dataset = Kinetics400Dataset(train=True, frames_per_clip=30, total_clips=10)
    _validate_kinetics(train_dataset, 112)

    val_dataset = Kinetics400Dataset(train=False, frames_per_clip=30, total_clips=10)
    _validate_kinetics(val_dataset, 112)

    reg_dataset = DatasetRegistry.create(
        "kinetics400", train=False, frames_per_clip=30, total_clips=10
    )
    _validate_kinetics(reg_dataset, 112)
