import os
import tempfile

import pytest
from sparseml.pytorch.datasets import DatasetRegistry, MNISTDataset
from torch.utils.data import Dataset


def _validate_mnist(dataset: Dataset):
    assert len(dataset) > 0
    item = dataset[0]
    assert isinstance(item, tuple)
    assert item[0].shape[0] == 1
    assert item[0].shape[1] == 28
    assert item[0].shape[2] == 28
    assert item[1] < 10


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_DATASET_TESTS", False),
    reason="Skipping dataset tests",
)
def test_mnist():
    train_dataset = MNISTDataset(train=True)
    _validate_mnist(train_dataset)

    val_dataset = MNISTDataset(train=False)
    _validate_mnist(val_dataset)

    reg_dataset = DatasetRegistry.create("mnist", train=False)
    _validate_mnist(reg_dataset)
