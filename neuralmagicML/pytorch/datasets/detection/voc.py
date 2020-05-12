"""
VOC dataset implementations for the object detection field in computer vision.
More info for the dataset can be found
`here <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/
PascalVOC_IJCV2009.pdf>`__.
"""

import os
from torchvision import transforms

try:
    from torchvision.datasets import VOCSegmentation, VOCDetection
except ModuleNotFoundError:
    # older version of pytorch, VOC not available
    VOCSegmentation = object
    VOCDetection = object

from neuralmagicML.pytorch.datasets.registry import DatasetRegistry
from neuralmagicML.utils.datasets import (
    default_dataset_path,
    IMAGENET_RGB_MEANS,
    IMAGENET_RGB_STDS,
)


__all__ = ["VOCSegmentationDataset", "VOCDetectionDataset"]


@DatasetRegistry.register(
    key=["voc_seg", "voc_segmentation"],
    attributes={
        "num_classes": 21,
        "transform_means": IMAGENET_RGB_MEANS,
        "transform_stds": IMAGENET_RGB_STDS,
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
    :param year: The dataset year, supports years 2007 to 2012.
    :param image_size: the size of the image to output from the dataset
    """

    def __init__(
        self,
        root: str = default_dataset_path("voc-segmentation"),
        train: bool = True,
        rand_trans: bool = False,
        download: bool = True,
        year: str = "2012",
        image_size: int = 300,
    ):
        if VOCSegmentation == object:
            raise ValueError(
                "VOC is unsupported on this PyTorch version, please upgrade to use"
            )

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
                transforms.Normalize(mean=IMAGENET_RGB_MEANS, std=IMAGENET_RGB_STDS),
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
        "transform_means": IMAGENET_RGB_MEANS,
        "transform_stds": IMAGENET_RGB_STDS,
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
        root: str = default_dataset_path("voc-detection"),
        train: bool = True,
        rand_trans: bool = False,
        download: bool = True,
        year: str = "2012",
        image_size: int = 300,
    ):
        if VOCDetection == object:
            raise ValueError(
                "VOC is unsupported on this PyTorch version, please upgrade to use"
            )

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
                transforms.Normalize(mean=IMAGENET_RGB_MEANS, std=IMAGENET_RGB_STDS),
            ]
        )

        super().__init__(
            root,
            year=year,
            image_set="train" if train else "val",
            download=download,
            transform=transforms.Compose(trans),
        )
