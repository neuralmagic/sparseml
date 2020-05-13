import os
import pytest

from torch.utils.data import Dataset

from neuralmagicML.pytorch.datasets import (
    DatasetRegistry,
    CocoDetectionDataset,
)


def _validate_coco(dataset: Dataset, size: int):
    assert len(dataset) > 0
    item = dataset[0]
    assert isinstance(item, tuple)
    assert item[0].shape[0] == 3
    assert item[0].shape[1] == size
    assert item[0].shape[2] == size
    assert len(item[1]) > 0


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_DATASET_TESTS", False), reason="Skipping dataset tests",
)
def test_coco_detection():
    # TODO: disabling for 1.0 release
    pass
    # 18 GB download
    # train_dataset = CocoDetectionDataset(train=True)
    # _validate_coco(train_dataset, 300)

    # val_dataset = CocoDetectionDataset(train=False)
    # _validate_coco(val_dataset, 300)
    #
    # reg_dataset = DatasetRegistry.create("coco", train=False)
    # _validate_coco(reg_dataset, 300)
