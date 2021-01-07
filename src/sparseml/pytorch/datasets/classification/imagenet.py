"""
Imagenet dataset implementations for the image classification field in computer vision.
More info for the dataset can be found `here <http://www.image-net.org/>`__.
"""

import os
import random

from sparseml.pytorch.datasets.registry import DatasetRegistry
from sparseml.utils import clean_path
from sparseml.utils.datasets import (
    IMAGENET_RGB_MEANS,
    IMAGENET_RGB_STDS,
    default_dataset_path,
)
from torchvision import transforms
from torchvision.datasets import ImageFolder


__all__ = ["ImageNetDataset"]


@DatasetRegistry.register(
    key=["imagenet"],
    attributes={
        "num_classes": 1000,
        "transform_means": IMAGENET_RGB_MEANS,
        "transform_stds": IMAGENET_RGB_STDS,
    },
)
class ImageNetDataset(ImageFolder):
    """
    Wrapper for the ImageNet dataset to apply standard transforms.

    :param root: The root folder to find the dataset at
    :param train: True if this is for the training distribution,
        False for the validation
    :param rand_trans: True to apply RandomCrop and RandomHorizontalFlip to the data,
        False otherwise
    :param image_size: the size of the image to output from the dataset
    """

    def __init__(
        self,
        root: str = default_dataset_path("imagenet"),
        train: bool = True,
        rand_trans: bool = False,
        image_size: int = 224,
    ):
        root = clean_path(root)
        non_rand_resize_scale = 256.0 / 224.0  # standard used
        init_trans = (
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
            ]
            if rand_trans
            else [
                transforms.Resize(round(non_rand_resize_scale * image_size)),
                transforms.CenterCrop(image_size),
            ]
        )

        trans = [
            *init_trans,
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_RGB_MEANS, std=IMAGENET_RGB_STDS),
        ]
        root = os.path.join(
            os.path.abspath(os.path.expanduser(root)), "train" if train else "val"
        )

        super().__init__(root, transform=transforms.Compose(trans))

        if train:
            # make sure we don't preserve the folder structure class order
            random.shuffle(self.samples)
