# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Imagenet dataset implementations for the image classification field in computer vision.
More info for the dataset can be found `here <http://www.image-net.org/>`__.
"""

import os
import random

from sparseml.pytorch.datasets.image_classification.ffcv_dataset import (
    FFCVImageNetDataset,
)


try:
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

except Exception as torchvision_error:
    raise ImportError(
        "torchvision dependencies not found, kindly install using "
        f"`pip install sparseml[torchvision]`, {torchvision_error}"
    )

from sparseml.pytorch.datasets.registry import DatasetRegistry
from sparseml.utils import clean_path
from sparseml.utils.datasets import (
    IMAGENET_RGB_MEANS,
    IMAGENET_RGB_STDS,
    default_dataset_path,
)


__all__ = ["ImageFolderDataset"]


@DatasetRegistry.register(
    key=["imagefolder"],
    attributes={
        "transform_means": IMAGENET_RGB_MEANS,
        "transform_stds": IMAGENET_RGB_STDS,
    },
)
class ImageFolderDataset(ImageFolder, FFCVImageNetDataset):
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
    | root/cat/asd932.png

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
            # make sure we dont preserve the folder structure class order
            random.shuffle(self.samples)

    @property
    def num_classes(self) -> int:
        return len(self.classes)
