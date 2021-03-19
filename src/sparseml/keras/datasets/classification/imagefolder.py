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
ImageFolder dataset implementations for the image
classification field in computer vision.
"""

import glob
import os
import random
from typing import Callable, Iterable, NamedTuple, Tuple, Union

import numpy
import tensorflow

from sparseml.keras.datasets.dataset import Dataset
from sparseml.keras.datasets.helpers import random_scaling_crop
from sparseml.keras.datasets.registry import DatasetRegistry
from sparseml.keras.utils.compat import keras
from sparseml.utils import clean_path
from sparseml.utils.datasets import IMAGENET_RGB_MEANS, IMAGENET_RGB_STDS


__all__ = ["imagenet_normalizer", "ImageFolderDataset", "SplitsTransforms"]


SplitsTransforms = NamedTuple(
    "SplitsTransforms",
    [
        ("train", Union[Iterable[Callable], None]),
        ("val", Union[Iterable[Callable], None]),
    ],
)


def imagenet_normalizer(img: tensorflow.Tensor, mode: str):
    """
    Normalize an image using mean and std of the imagenet dataset
    :param img: The input image to normalize
    :param mode: either "tf", "caffe", "torch"
    :return: The normalized image
    """
    if mode == "tf":
        preprocess_input = keras.applications.mobilenet.preprocess_input
    elif mode == "caffe":
        preprocess_input = keras.applications.resnet.preprocess_input
    elif mode == "torch":
        preprocess_input = None
    else:
        raise ValueError("Unknown preprocessing method")
    if preprocess_input is not None:
        processed_image = preprocess_input(img)
    else:
        res = tensorflow.cast(img, dtype=tensorflow.float32) / 255.0
        means = tensorflow.constant(IMAGENET_RGB_MEANS, dtype=tensorflow.float32)
        stds = tensorflow.constant(IMAGENET_RGB_STDS, dtype=tensorflow.float32)
        processed_image = (res - means) / stds
    return processed_image


def default_imagenet_normalizer():
    def normalizer(img: tensorflow.Tensor):
        # Default to the same preprocessing used by Keras Applications ResNet
        return imagenet_normalizer(img, "caffe")

    return normalizer


@DatasetRegistry.register(
    key=["imagefolder"],
    attributes={
        "transform_means": IMAGENET_RGB_MEANS,
        "transform_stds": IMAGENET_RGB_STDS,
    },
)
class ImageFolderDataset(Dataset):
    """
    Implementation for loading an image folder structure into a dataset.
    | Image folders should be of the form:
    |   root/class_x/xxx.ext
    |   root/class_x/xxy.ext
    |   root/class_x/xxz.ext
    |
    |   root/class_y/123.ext
    |   root/class_y/nsdf3.ext
    |   root/class_y/asd932_.ext
    :param root: the root location for the dataset's images to load
    :param train: True to load the training dataset from the root,
        False for validation
    :param image_size: the size of the image to reshape to
    :param pre_resize_transforms: transforms to be applied before resizing the image
    :param post_resize_transforms: transforms to be applied after resizing the image
    """

    def __init__(
        self,
        root: str,
        train: bool,
        image_size: Union[None, int, Tuple[int, int]] = 224,
        pre_resize_transforms: Union[SplitsTransforms, None] = SplitsTransforms(
            train=(
                random_scaling_crop(),
                tensorflow.image.random_flip_left_right,
            ),
            val=None,
        ),
        post_resize_transforms: Union[SplitsTransforms, None] = SplitsTransforms(
            train=(default_imagenet_normalizer(),),
            val=(default_imagenet_normalizer(),),
        ),
    ):
        self._root = os.path.join(clean_path(root), "train" if train else "val")
        if not os.path.exists(self._root):
            raise ValueError("Data set folder {} must exist".format(self._root))
        self._train = train
        if image_size is not None:
            self._image_size = (
                image_size
                if isinstance(image_size, tuple)
                else (image_size, image_size)
            )
        else:
            self._image_size = None
        self._pre_resize_transforms = pre_resize_transforms
        self._post_resize_transforms = post_resize_transforms

        self._num_images = len(
            [None for _ in glob.glob(os.path.join(self._root, "*", "*"))]
        )
        self._num_classes = len(
            [None for _ in glob.glob(os.path.join(self._root, "*", ""))]
        )

    def __len__(self):
        return self._num_images

    @property
    def root(self) -> str:
        """
        :return: the root location for the dataset's images to load
        """
        return self._root

    @property
    def train(self) -> bool:
        """
        :return: True to load the training dataset from the root, False for validation
        """
        return self._train

    @property
    def image_size(self) -> Tuple[int, int]:
        """
        :return: the size of the images to resize to
        """
        return self._image_size

    @property
    def pre_resize_transforms(self) -> SplitsTransforms:
        """
        :return: transforms to be applied before resizing the image
        """
        return self._pre_resize_transforms

    @property
    def post_resize_transforms(self) -> SplitsTransforms:
        """
        :return: transforms to be applied after resizing the image
        """
        return self._post_resize_transforms

    @property
    def num_images(self) -> int:
        """
        :return: the number of images found for the dataset
        """
        return self._num_images

    @property
    def num_classes(self):
        """
        :return: the number of classes found for the dataset
        """
        return self._num_classes

    def processor(self, file_path: tensorflow.Tensor, label: tensorflow.Tensor):
        """
        :param file_path: the path to the file to load an image from
        :param label: the label for the given image
        :return: a tuple containing the processed image and label
        """
        img = tensorflow.io.read_file(file_path)
        img = tensorflow.image.decode_jpeg(img, channels=3)
        if self.pre_resize_transforms:
            transforms = (
                self.pre_resize_transforms.train
                if self.train
                else self.pre_resize_transforms.val
            )
            if transforms:
                for trans in transforms:
                    img = trans(img)
        if self._image_size is not None:
            img = tensorflow.image.resize(img, self.image_size)

        if self.post_resize_transforms:
            transforms = (
                self.post_resize_transforms.train
                if self.train
                else self.post_resize_transforms.val
            )
            if transforms:
                for trans in transforms:
                    img = trans(img)
        return img, label

    def creator(self):
        """
        :return: a created dataset that gives the file_path and label for each
            image under self.root
        """
        labels_strs = [
            fold.split(os.path.sep)[-1]
            for fold in glob.glob(os.path.join(self.root, "*"))
        ]
        labels_strs.sort()
        labels_dict = {
            lab: numpy.identity(len(labels_strs))[index].tolist()
            for index, lab in enumerate(labels_strs)
        }
        files_labels = [
            (file, labels_dict[file.split(os.path.sep)[-2]])
            for file in glob.glob(os.path.join(self.root, "*", "*"))
        ]
        random.Random(42).shuffle(files_labels)
        files, labels = zip(*files_labels)
        files = tensorflow.constant(files)
        labels = tensorflow.constant(labels)

        return tensorflow.data.Dataset.from_tensor_slices((files, labels))
