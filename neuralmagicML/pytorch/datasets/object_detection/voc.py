"""
VOC dataset implementations for the object detection field in computer vision.
More info for the dataset can be found
`here <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/
PascalVOC_IJCV2009.pdf>`__.
"""

import os
from torchvision import transforms
from torchvision.datasets import VOCSegmentation, VOCDetection

from neuralmagicML.pytorch.datasets.registry import DatasetRegistry


__all__ = ["VOCSegmentationDataset", "VOCDetectionDataset"]


_RGB_MEANS = [0.485, 0.456, 0.406]
_RGB_STDS = [0.229, 0.224, 0.225]


@DatasetRegistry.register(
    key=["voc_seg", "voc_segmentation"],
    attributes={
        "num_classes": 21,
        "transform_means": _RGB_MEANS,
        "transform_stds": _RGB_STDS,
    },
)
class VOCSegmentationDataset(VOCSegmentation):
    """
    Wrapper for the VOC Segmentation dataset to apply standard transforms.

    :param root: The root folder to find the dataset at, if not found will
        download here if download=True
    :param train: True if this is for the training distribution,
        False for the validation
    :param rand_trans: True to apply RandomCrop and RandomHorizontalFlip to the data,
        False otherwise
    :param download: True to download the dataset, False otherwise.
    :param image_size: the size of the image to output from the dataset
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        rand_trans: bool = False,
        download: bool = True,
        year: str = "2012",
        image_size: int = 300,
    ):
        root = os.path.abspath(os.path.expanduser(root))
        trans = (
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
            ]
            if rand_trans
            else [transforms.Resize((image_size, image_size))]
        )
        trans.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=_RGB_MEANS, std=_RGB_STDS),
            ]
        )

        super().__init__(
            root,
            year=year,
            image_set="train" if train else "val",
            download=download,
            transform=transforms.Compose(trans),
        )


@DatasetRegistry.register(
    key=["voc_det", "voc_detection"],
    attributes={
        "num_classes": 21,
        "transform_means": _RGB_MEANS,
        "transform_stds": _RGB_STDS,
    },
)
class VOCDetectionDataset(VOCDetection):
    """
    Wrapper for the VOCDetection dataset to apply standard transforms.

    :param root: The root folder to find the dataset at, if not found will download
        here if download=True
    :param train: True if this is for the training distribution,
        False for the validation
    :param rand_trans: True to apply RandomCrop and RandomHorizontalFlip to the data,
        False otherwise
    :param download: True to download the dataset, False otherwise.
        Base implementation does not support leaving as false if already downloaded
    :param image_size: the size of the image to output from the dataset
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        rand_trans: bool = False,
        download: bool = True,
        year: str = "2012",
        image_size: int = 300,
    ):
        root = os.path.abspath(os.path.expanduser(root))
        trans = (
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
            ]
            if rand_trans
            else [transforms.Resize((image_size, image_size))]
        )
        trans.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=_RGB_MEANS, std=_RGB_STDS),
            ]
        )

        super().__init__(
            root,
            year=year,
            image_set="train" if train else "val",
            download=download,
            transform=transforms.Compose(trans),
        )
