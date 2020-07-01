"""
Imagenet dataset implementations for the image classification field in computer vision.
More info for the dataset can be found `here <http://www.image-net.org/>`__.
"""

import os
import random
import PIL.Image as Image

from torchvision import transforms
from torchvision.datasets import ImageFolder

from neuralmagicML.utils import clean_path
from neuralmagicML.pytorch.datasets.registry import DatasetRegistry
from neuralmagicML.utils.datasets import (
    default_dataset_path,
    IMAGENET_RGB_MEANS,
    IMAGENET_RGB_STDS,
)


__all__ = ["ImageFolderDataset"]


@DatasetRegistry.register(
    key=["imagefolder"],
    attributes={
        "transform_means": IMAGENET_RGB_MEANS,
        "transform_stds": IMAGENET_RGB_STDS,
    },
)
class ImageFolderDataset(ImageFolder):
    """
    Wrapper for the ImageFolder dataset to apply standard transforms.
    Additionally scales the inputs based off of the imagenet means and stds.

    | Dataset should be in the following form locally on disk:
    |
    | root/dog/xxx.png
    | root/dog/xxy.png
    | root/dog/xxz.png
    |
    | root/cat/123.png
    | root/cat/nsdf3.png
    | root/cat/asd932_.png

    :param root: The root folder to find the dataset at
    :param train: True if this is for the training distribution,
        False for the validation
    :param rand_trans: True to apply RandomCrop and RandomHorizontalFlip to the data,
        False otherwise
    :param image_size: the size of the image to output from the dataset
    """

    def __init__(
        self,
        root: str = default_dataset_path("imagenet"),  # default to imagenet location
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

    @property
    def num_classes(self) -> int:
        return len(self.classes)
