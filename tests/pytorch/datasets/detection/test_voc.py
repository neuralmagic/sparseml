import os
import pytest
from packaging import version

import torch
from torch.utils.data import Dataset

from neuralmagicML.pytorch.datasets import (
    DatasetRegistry,
    VOCSegmentationDataset,
    VOCDetectionDataset,
)


def _validate_voc(dataset: Dataset, size: int):
    assert len(dataset) > 0
    item = dataset[0]
    assert isinstance(item, tuple)
    assert item[0].shape[0] == 3
    assert item[0].shape[1] == size
    assert item[0].shape[2] == size
    assert len(item[1]) == 2


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("1.2"),
    reason="Must install pytorch version 1.2 or greater",
)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_DATASET_TESTS", False), reason="Skipping dataset tests",
)
def test_voc_detection():
    train_dataset = VOCDetectionDataset(train=True)
    _validate_voc(train_dataset, 300)

    val_dataset = VOCDetectionDataset(train=False)
    _validate_voc(val_dataset, 300)

    reg_dataset = DatasetRegistry.create("voc_det", train=False)
    _validate_voc(reg_dataset, 300)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("1.2"),
    reason="Must install pytorch version 1.2 or greater",
)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_DATASET_TESTS", False), reason="Skipping dataset tests",
)
def test_voc_segmentation():
    pass
