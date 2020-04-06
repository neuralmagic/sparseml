import os
import tempfile
from torch.utils.data import Dataset

from neuralmagicML.pytorch.datasets import (
    DatasetRegistry,
    ImagewoofDataset,
    ImagewoofSize,
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
    datasets_path = os.path.join(tempfile.gettempdir(), "datasets")
    train_dataset = ImagewoofDataset(datasets_path, train=True)
    _validate_imagewoof(train_dataset, 160)

    val_dataset = ImagewoofDataset(datasets_path, train=False)
    _validate_imagewoof(val_dataset, 160)

    reg_dataset = DatasetRegistry.create("imagewoof", datasets_path, train=False)
    _validate_imagewoof(reg_dataset, 160)


def test_imagewoof_320():
    datasets_path = os.path.join(tempfile.gettempdir(), "datasets")
    train_dataset = ImagewoofDataset(
        datasets_path, train=True, dataset_size=ImagewoofSize.s320
    )
    _validate_imagewoof(train_dataset, 224)

    val_dataset = ImagewoofDataset(
        datasets_path, train=False, dataset_size=ImagewoofSize.s320
    )
    _validate_imagewoof(val_dataset, 224)

    reg_dataset = DatasetRegistry.create(
        "imagewoof", datasets_path, train=False, dataset_size=ImagewoofSize.s320
    )
    _validate_imagewoof(reg_dataset, 224)


def test_imagewoof_full():
    datasets_path = os.path.join(tempfile.gettempdir(), "datasets")
    train_dataset = ImagewoofDataset(
        datasets_path, train=True, dataset_size=ImagewoofSize.full
    )
    _validate_imagewoof(train_dataset, 224)

    val_dataset = ImagewoofDataset(
        datasets_path, train=False, dataset_size=ImagewoofSize.full
    )
    _validate_imagewoof(val_dataset, 224)

    reg_dataset = DatasetRegistry.create(
        "imagewoof", datasets_path, train=False, dataset_size=ImagewoofSize.full
    )
    _validate_imagewoof(reg_dataset, 224)
