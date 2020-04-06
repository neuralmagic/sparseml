import os
import tempfile
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
    datasets_path = os.path.join(tempfile.gettempdir(), "datasets")
    train_dataset = ImagenetteDataset(datasets_path, train=True)
    _validate_imagenette(train_dataset, 160)

    val_dataset = ImagenetteDataset(datasets_path, train=False)
    _validate_imagenette(val_dataset, 160)

    reg_dataset = DatasetRegistry.create("imagenette", datasets_path, train=False)
    _validate_imagenette(reg_dataset, 160)


def test_imagenette_320():
    datasets_path = os.path.join(tempfile.gettempdir(), "datasets")
    train_dataset = ImagenetteDataset(
        datasets_path, train=True, dataset_size=ImagenetteSize.s320
    )
    _validate_imagenette(train_dataset, 224)

    val_dataset = ImagenetteDataset(
        datasets_path, train=False, dataset_size=ImagenetteSize.s320
    )
    _validate_imagenette(val_dataset, 224)

    reg_dataset = DatasetRegistry.create(
        "imagenette", datasets_path, train=False, dataset_size=ImagenetteSize.s320
    )
    _validate_imagenette(reg_dataset, 224)


def test_imagenette_full():
    datasets_path = os.path.join(tempfile.gettempdir(), "datasets")
    train_dataset = ImagenetteDataset(
        datasets_path, train=True, dataset_size=ImagenetteSize.full
    )
    _validate_imagenette(train_dataset, 224)

    val_dataset = ImagenetteDataset(
        datasets_path, train=False, dataset_size=ImagenetteSize.full
    )
    _validate_imagenette(val_dataset, 224)

    reg_dataset = DatasetRegistry.create(
        "imagenette", datasets_path, train=False, dataset_size=ImagenetteSize.full
    )
    _validate_imagenette(reg_dataset, 224)
