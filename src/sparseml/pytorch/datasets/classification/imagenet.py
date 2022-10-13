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


try:
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    torchvision_import_error = None
except Exception as torchvision_error:
    transforms = None
    ImageFolder = object  # default for constructor
    torchvision_import_error = torchvision_error

from typing import List, Union

from sparseml.pytorch.datasets.image_classification.ffcv_dataset import (
    FFCVImageNetDataset,
)
from sparseml.pytorch.datasets.registry import DatasetRegistry
from sparseml.utils import clean_path
from sparseml.utils.datasets import (
    IMAGENET_RGB_MEANS,
    IMAGENET_RGB_STDS,
    default_dataset_path,
)


__all__ = ["ImageNetDataset"]


@DatasetRegistry.register(
    key=["imagenet"],
    attributes={
        "num_classes": 1000,
        "transform_means": IMAGENET_RGB_MEANS,
        "transform_stds": IMAGENET_RGB_STDS,
    },
)
class ImageNetDataset(ImageFolder, FFCVImageNetDataset):
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
        resize_scale: float = 1.143,
        resize_mode: Union[str, "transforms.InterpolationMode"] = "bilinear",
        rgb_means: List[float] = IMAGENET_RGB_MEANS,
        rgb_stds: List[float] = IMAGENET_RGB_STDS,
    ):
        if torchvision_import_error is not None:
            raise torchvision_import_error

        root = clean_path(root)
        if type(resize_mode) is str and resize_mode.lower() in ["linear", "bilinear"]:
            interpolation = transforms.InterpolationMode.BILINEAR
        elif type(resize_mode) is str and resize_mode.lower() in ["cubic", "bicubic"]:
            interpolation = transforms.InterpolationMode.BICUBIC

        init_trans = (
            [
                transforms.RandomResizedCrop(image_size, interpolation=interpolation),
                transforms.RandomHorizontalFlip(),
            ]
            if rand_trans
            else [
                transforms.Resize(
                    round(resize_scale * image_size), interpolation=interpolation
                ),
                transforms.CenterCrop(image_size),
            ]
        )

        trans = [
            *init_trans,
            transforms.ToTensor(),
            transforms.Normalize(mean=rgb_means, std=rgb_stds),
        ]
        root = os.path.join(
            os.path.abspath(os.path.expanduser(root)), "train" if train else "val"
        )

        super().__init__(root, transform=transforms.Compose(trans))
        self.image_size = image_size
        self.rand_trans = rand_trans

        if train:
            # make sure we don't preserve the folder structure class order
            random.shuffle(self.samples)
