import os

import pytest
from torch.utils.data import DataLoader, Dataset


try:
    import pycocotools
except:
    pycocotools = None

from sparseml.pytorch.datasets import CocoDetectionDataset, DatasetRegistry


def _validate_coco(dataset: Dataset, size: int):
    assert len(dataset) > 0
    item = dataset[0]
    assert isinstance(item, tuple)
    assert item[0].shape[0] == 3
    assert item[0].shape[1] == size
    assert item[0].shape[2] == size
    assert len(item[1]) == 2


@pytest.mark.skipif(pycocotools is None, reason="Pycocotools not installed")
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_DATASET_TESTS", False),
    reason="Skipping dataset tests",
)
def test_coco_detection():
    # 18 GB download
    train_dataset = CocoDetectionDataset(train=True)
    _validate_coco(train_dataset, 300)

    val_dataset = CocoDetectionDataset(train=False)
    _validate_coco(val_dataset, 300)

    reg_dataset = DatasetRegistry.create("coco", train=False)
    _validate_coco(reg_dataset, 300)
