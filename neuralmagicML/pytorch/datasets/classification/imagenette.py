"""
Imagenette dataset implementations for the image classification field in
computer vision.
More info for the dataset can be found `here <https://github.com/fastai/imagenette>`__.
"""

from typing import Union
import os
from enum import Enum
import random
import tarfile
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url

from neuralmagicML.pytorch.datasets.registry import DatasetRegistry
from neuralmagicML.pytorch.datasets.generic import default_dataset_path


__all__ = ["ImagenetteSize", "ImagenetteDataset"]


_RGB_MEANS = [0.485, 0.456, 0.406]
_RGB_STDS = [0.229, 0.224, 0.225]


class ImagenetteSize(Enum):
    """
    Dataset size for Imagenette.
    full does not resize the original dataset at all.
    s320 resizes the images to 320px.
    s160 resizes the images to 160px.
    """

    full = "full"
    s320 = "s320"
    s160 = "s160"


@DatasetRegistry.register(
    key=["imagenette"],
    attributes={
        "num_classes": 1000,
        "transform_means": _RGB_MEANS,
        "transform_stds": _RGB_STDS,
    },
)
class ImagenetteDataset(ImageFolder):
    """
    Wrapper for the imagenette (10 class) dataset that fastai created.
    Handles downloading and applying standard transforms.

    :param root: The root folder to find the dataset at,
        if not found will download here if download=True
    :param train: True if this is for the training distribution,
        False for the validation
    :param rand_trans: True to apply RandomCrop and RandomHorizontalFlip to the data,
        False otherwise
    :param dataset_size: The size of the dataset to use and download:
        See ImagenetteSize for options
    :param image_size: The image size to output from the dataset
    :param download: True to download the dataset, False otherwise
    """

    def __init__(
        self,
        root: str = default_dataset_path("imagenette"),
        train: bool = True,
        rand_trans: bool = False,
        dataset_size: ImagenetteSize = ImagenetteSize.s160,
        image_size: Union[int, None] = None,
        download: bool = True,
    ):
        root = os.path.abspath(os.path.expanduser(root))

        if not os.path.exists(root):
            os.makedirs(root)

        extracted_root = root

        if download:
            if dataset_size == ImagenetteSize.s160:
                extract = "imagenette-160"
            elif dataset_size == ImagenetteSize.s320:
                extract = "imagenette-320"
            elif dataset_size == ImagenetteSize.full:
                extract = "imagenette"
            else:
                raise ValueError(
                    "Unknown ImagenetteSize given of {}".format(dataset_size)
                )

            extracted_root = os.path.join(root, extract)
            ImagenetteDataset._download(dataset_size, root, extract)

        if image_size is None:
            image_size = 160 if dataset_size == ImagenetteSize.s160 else 224

        extracted_root = (
            os.path.join(extracted_root, "train")
            if train
            else os.path.join(extracted_root, "val")
        )

        if rand_trans:
            trans = [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            resize_scale = 256.0 / 224.0  # standard used
            trans = [
                transforms.Resize(round(resize_scale * image_size)),
                transforms.CenterCrop(image_size),
            ]

        trans.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=_RGB_MEANS, std=_RGB_STDS),
            ]
        )

        super().__init__(extracted_root, transforms.Compose(trans))

        # make sure we don't preserve the folder structure class order
        random.shuffle(self.samples)

    @staticmethod
    def _download(size: ImagenetteSize, root: str, extract: str):
        if size == ImagenetteSize.full:
            url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette.tgz"
        elif size == ImagenetteSize.s320:
            url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette-320.tgz"
        elif size == ImagenetteSize.s160:
            url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette-160.tgz"
        else:
            raise ValueError("unknown imagenette size given of {}".format(size))

        filename = "{}.tar".format(extract)

        if os.path.exists(os.path.join(root, filename)):
            print("already downloaded imagenette of size {}".format(size))

            return

        download_url(url, root, filename)

        with tarfile.open(os.path.join(root, filename), "r:gz") as tar:
            tar.extractall(path=root)
