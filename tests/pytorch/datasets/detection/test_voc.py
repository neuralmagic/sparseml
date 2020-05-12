import os
import pytest

import torch
from torch.utils.data import Dataset

from neuralmagicML.pytorch.datasets import (
    DatasetRegistry,
    VOCSegmentationDataset,
    VOCDetectionDataset,
)


def pytorch_version_valid():
    pytorch_version = [int(version) for version in torch.__version__.split(".")[:2]]
    return pytorch_version[0] > 1 or (
        pytorch_version[0] == 1 and pytorch_version[1] >= 2
    )


def _validate_voc(dataset: Dataset, size: int):
    assert len(dataset) > 0
    item = dataset[0]
    assert isinstance(item, tuple)
    assert item[0].shape[0] == 3
    assert item[0].shape[1] == size
    assert item[0].shape[2] == size
    assert len(item[1]) > 0


@pytest.mark.skipif(
    not pytorch_version_valid(), reason="Must install pytorch version 1.2 or greater"
)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_DATASET_TESTS", False),
    reason="Skipping dataset tests",
)
def test_voc_detection():
    # TODO: disabling for 1.0 release
    pass
    # train_dataset = VOCDetectionDataset(train=True)
    # _validate_voc(train_dataset, 300)

    # val_dataset = VOCDetectionDataset(train=False)
    # _validate_voc(val_dataset, 300)

    # reg_dataset = DatasetRegistry.create("voc_det", train=False)
    # _validate_voc(reg_dataset, 300)


@pytest.mark.skipif(
    not pytorch_version_valid(), reason="Must install pytorch version 1.2 or greater"
)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_DATASET_TESTS", False),
    reason="Skipping dataset tests",
)
def test_voc_segmentation():
    pass
