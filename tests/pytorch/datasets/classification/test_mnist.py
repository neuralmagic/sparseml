import os
import tempfile
from torch.utils.data import Dataset

from neuralmagicML.pytorch.datasets import DatasetRegistry, MNISTDataset


def _validate_mnist(dataset: Dataset):
    assert len(dataset) > 0
    item = dataset[0]
    assert isinstance(item, tuple)
    assert item[0].shape[0] == 1
    assert item[0].shape[1] == 28
    assert item[0].shape[2] == 28
    assert item[1] < 10


def test_mnist():
    datasets_path = os.path.join(tempfile.gettempdir(), "datasets")
    train_dataset = MNISTDataset(datasets_path, train=True)
    _validate_mnist(train_dataset)

    val_dataset = MNISTDataset(datasets_path, train=False)
    _validate_mnist(val_dataset)

    reg_dataset = DatasetRegistry.create("mnist", datasets_path, train=False)
    _validate_mnist(reg_dataset)
