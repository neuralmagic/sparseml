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

import random
from typing import Tuple, Union

import tensorflow as tf

from sparseml.keras.datasets.classification import (
    ImageFolderDataset,
    SplitsTransforms,
    imagenet_normalizer,
)
from sparseml.keras.datasets.helpers import random_scaling_crop
from sparseml.keras.datasets.registry import DatasetRegistry
from sparseml.keras.utils import keras
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


def imagenet_pre_resize_processor():
    def processor(image: tf.Tensor):
        image_batch = tf.expand_dims(image, axis=0)

        # Resize the image the following way to match torchvision's Resize
        # transform used by Pytorch code path for Imagenet:
        #   torchvision.transforms.Resize(256)
        # which resize the smaller side of images to 256 and the other one based
        # on the aspect ratio
        shape = tf.shape(image)
        h, w = shape[0], shape[1]
        if h > w:
            new_h, new_w = tf.cast(256 * h / w, dtype=tf.uint16), tf.constant(
                256, dtype=tf.uint16
            )
        else:
            new_h, new_w = tf.constant(256, dtype=tf.uint16), tf.cast(
                256 * w / h, dtype=tf.uint16
            )
        resizer = keras.layers.experimental.preprocessing.Resizing(new_h, new_w)
        image_batch = tf.cast(resizer(image_batch), dtype=tf.uint8)

        # Center crop
        center_cropper = keras.layers.experimental.preprocessing.CenterCrop(224, 224)
        image_batch = tf.cast(center_cropper(image_batch), dtype=tf.uint8)

        return image_batch[0, :]

    return processor


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
        image_size: Union[None, int, Tuple[int, int]] = 224,
        pre_resize_transforms=SplitsTransforms(
            train=(
                random_scaling_crop(),
                tf.image.random_flip_left_right,
            ),
            val=(imagenet_pre_resize_processor(),),
        ),
        post_resize_transforms=SplitsTransforms(
            train=(torch_imagenet_normalizer(),), val=(torch_imagenet_normalizer(),)
        ),
    ):
        root = clean_path(root)
        super().__init__(
            root,
            train,
            image_size=image_size,
            pre_resize_transforms=pre_resize_transforms,
            post_resize_transforms=post_resize_transforms,
        )

        if train:
            # make sure we don't preserve the folder structure class order
            random.shuffle(self.samples)
