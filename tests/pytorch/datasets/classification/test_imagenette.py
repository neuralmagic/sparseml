from torch.utils.data import Dataset

from neuralmagicML.pytorch.datasets import (
    DatasetRegistry,
    ImagenetteDataset,
    ImagenetteSize,
)


def _validate_imagenette(dataset: Dataset, size: int):
    assert len(dataset) > 0
    item = dataset[0]
    assert isinstance(item, tuple)
    assert item[0].shape[0] == 3
    assert item[0].shape[1] == size
    assert item[0].shape[2] == size
    assert item[1] < 10


def test_imagenette_160():
    train_dataset = ImagenetteDataset(train=True)
    _validate_imagenette(train_dataset, 160)

    val_dataset = ImagenetteDataset(train=False)
    _validate_imagenette(val_dataset, 160)

    reg_dataset = DatasetRegistry.create("imagenette", train=False)
    _validate_imagenette(reg_dataset, 160)
