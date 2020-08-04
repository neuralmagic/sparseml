"""
VOC dataset implementations for the object detection field in computer vision.
More info for the dataset can be found
`here <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/
PascalVOC_IJCV2009.pdf>`__.
"""

import os
from PIL import Image
import random
from torchvision import transforms
from torchvision.transforms import functional as F

try:
    from torchvision.datasets import VOCSegmentation, VOCDetection
except ModuleNotFoundError:
    # older version of pytorch, VOC not available
    VOCSegmentation = object
    VOCDetection = object

from neuralmagicML.pytorch.datasets.detection.helpers import AnnotatedImageTransforms
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
        trans = [lambda img, ann: _resize_detection(img, ann, image_size)]
        if rand_trans:
            trans.append(lambda img, ann: _random_horizontal_flip_detection(img, ann))
        trans.extend(
            [
                # Convert to tensor
                lambda img, ann: (F.to_tensor(img), ann),
                # Normalize image
                lambda img, ann: (
                    F.normalize(img, IMAGENET_RGB_MEANS, IMAGENET_RGB_STDS),
                    ann,
                ),
            ]
        )

        super().__init__(
            root,
            year=year,
            image_set="train" if train else "val",
            download=download,
            transforms=AnnotatedImageTransforms(trans),
        )


def _resize_detection(image, annotations, image_size):
    if not isinstance(image, Image.Image):
        raise RuntimeError(
            "Loaded image class not supported, expected PIL.Image, got {}".format(
                image.__class__
            )
        )
    width_scale = image_size / image.size[0]
    height_scale = image_size / image.size[1]

    def update_bndbox_fn(bndbox):
        bndbox["xmin"] = float(bndbox["xmin"]) * width_scale
        bndbox["ymin"] = float(bndbox["ymin"]) * height_scale
        bndbox["xmax"] = float(bndbox["xmax"]) * width_scale
        bndbox["ymax"] = float(bndbox["ymax"]) * height_scale
        return bndbox

    image = F.resize(image, (image_size, image_size))
    annotations = _update_bndbox_values(annotations, update_bndbox_fn)

    return image, annotations


def _random_horizontal_flip_detection(image, annotations, p=0.5):
    if not isinstance(image, Image.Image):
        raise RuntimeError(
            "Loaded image class not supported, expected PIL.Image, got {}".format(
                image.__class__
            )
        )

    if random.random() < p:

        def bndbox_horizontal_flip_fn(bndbox):
            bndbox["xmin"] = image.size[0] - bndbox["xmax"]
            bndbox["xmax"] = image.size[0] - bndbox["xmin"]
            return bndbox

        image = F.hflip(image)
        annotations = _update_bndbox_values(annotations, bndbox_horizontal_flip_fn)

    return image, annotations


def _update_bndbox_values(annotations, update_bndbox_fn):
    for idx, annotation in enumerate(annotations["annotation"]["object"]):
        updated_bndbox = update_bndbox_fn(annotation["bndbox"])
        annotations["annotation"]["object"][idx]["bndbox"] = updated_bndbox
    return annotations
