from torch.utils.data import Dataset

from neuralmagicML.pytorch.datasets import (
    DatasetRegistry,
    ImagewoofDataset,
)


def _validate_imagewoof(dataset: Dataset, size: int):
    assert len(dataset) > 0
    item = dataset[0]
    assert isinstance(item, tuple)
    assert item[0].shape[0] == 3
    assert item[0].shape[1] == size
    assert item[0].shape[2] == size
    assert item[1] < 10


def test_imagewoof_160():
    train_dataset = ImagewoofDataset(train=True)
    _validate_imagewoof(train_dataset, 160)

    val_dataset = ImagewoofDataset(train=False)
    _validate_imagewoof(val_dataset, 160)

    reg_dataset = DatasetRegistry.create("imagewoof", train=False)
    _validate_imagewoof(reg_dataset, 160)
