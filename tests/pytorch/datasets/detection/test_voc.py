import os
import tempfile
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


def test_voc_segmentation_2007():
    datasets_path = os.path.join(tempfile.gettempdir(), "datasets")
    train_dataset = VOCSegmentationDataset(datasets_path, train=True, year="2007")
    _validate_voc(train_dataset, 300)

    val_dataset = VOCSegmentationDataset(datasets_path, train=False, year="2007")
    _validate_voc(val_dataset, 300)

    reg_dataset = DatasetRegistry.create(
        "voc_seg", datasets_path, year="2007", train=False
    )
    _validate_voc(reg_dataset, 300)


def test_voc_segmentation_2012():
    datasets_path = os.path.join(tempfile.gettempdir(), "datasets")
    train_dataset = VOCSegmentationDataset(datasets_path, train=True, year="2012")
    _validate_voc(train_dataset, 300)

    val_dataset = VOCSegmentationDataset(datasets_path, train=False, year="2012")
    _validate_voc(val_dataset, 300)

    reg_dataset = DatasetRegistry.create(
        "voc_seg", datasets_path, year="2012", train=False
    )
    _validate_voc(reg_dataset, 300)


def test_voc_detection_2007():
    datasets_path = os.path.join(tempfile.gettempdir(), "datasets")
    train_dataset = VOCDetectionDataset(datasets_path, train=True, year="2007")
    _validate_voc(train_dataset, 300)

    val_dataset = VOCDetectionDataset(datasets_path, train=False, year="2007")
    _validate_voc(val_dataset, 300)

    reg_dataset = DatasetRegistry.create(
        "voc_det", datasets_path, year="2007", train=False
    )
    _validate_voc(reg_dataset, 300)


def test_voc_detection_2012():
    datasets_path = os.path.join(tempfile.gettempdir(), "datasets")
    train_dataset = VOCDetectionDataset(datasets_path, train=True, year="2012")
    _validate_voc(train_dataset, 300)

    val_dataset = VOCDetectionDataset(datasets_path, train=False, year="2012")
    _validate_voc(val_dataset, 300)

    reg_dataset = DatasetRegistry.create(
        "voc_det", datasets_path, year="2012", train=False
    )
    _validate_voc(reg_dataset, 300)
