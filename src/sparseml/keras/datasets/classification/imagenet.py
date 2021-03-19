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

import tensorflow as tf

from sparseml.keras.datasets.classification import (
    ImageFolderDataset,
    SplitsTransforms,
    imagenet_normalizer,
)
from sparseml.keras.datasets.registry import DatasetRegistry
from sparseml.utils import clean_path
from sparseml.utils.datasets import (
    IMAGENET_RGB_MEANS,
    IMAGENET_RGB_STDS,
    default_dataset_path,
)


__all__ = ["ImageNetDataset"]


def torch_imagenet_normalizer():
    def normalizer(image: tf.Tensor):
        return imagenet_normalizer(image, "torch")

    return normalizer


@DatasetRegistry.register(
    key=["imagenet"],
    attributes={
        "num_classes": 1000,
        "transform_means": IMAGENET_RGB_MEANS,
        "transform_stds": IMAGENET_RGB_STDS,
    },
)
class ImageNetDataset(ImageFolderDataset):
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
        post_resize_transforms = SplitsTransforms(
            train=(torch_imagenet_normalizer(),), val=(torch_imagenet_normalizer(),)
        )
        super().__init__(root, train, post_resize_transforms=post_resize_transforms)

        if train:
            # make sure we don't preserve the folder structure class order
            random.shuffle(self.samples)
