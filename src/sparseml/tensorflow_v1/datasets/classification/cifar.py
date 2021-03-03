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

import os
import pickle
import tarfile
from typing import Union

import numpy as np
from PIL import Image
from tqdm import tqdm

from sparseml.tensorflow_v1.datasets.classification.imagefolder import (
    ImageFolderDataset,
    SplitsTransforms,
)
from sparseml.tensorflow_v1.datasets.registry import DatasetRegistry
from sparseml.tensorflow_v1.utils import tf_compat, tf_compat_div
from sparseml.utils import create_dirs
from sparsezoo.utils import download_file


__all__ = ["Cifar10DataSet", "Cifar100DataSet"]


_CIFAR_IMAGE_SIZE = 32
_PADDING = 4


def preprocess_for_train(image: tf_compat.Tensor):
    """
    The default preprocessing function for train set as defined in Resnet paper
    for Cifar datasets

    :param image: the image tensor

    :return: the preprocessed image
    """
    with tf_compat.name_scope("train_preprocess"):
        image = tf_compat.cast(image, dtype=tf_compat.float32)
        rand_choice = tf_compat.random_uniform(
            shape=[], minval=0, maxval=2, dtype=tf_compat.int32
        )
        padding = _PADDING
        image = tf_compat.cond(
            tf_compat.equal(rand_choice, 0),
            lambda: tf_compat.pad(
                image, [[padding, padding], [padding, padding], [0, 0]]
            ),
            lambda: tf_compat.image.random_flip_left_right(image),
        )
        distorted_image = tf_compat.image.random_crop(image, [32, 32, 3])
        return distorted_image


def preprocess_for_eval(image: tf_compat.Tensor):
    """
    The default preprocessing function for test set as defined in Resnet paper
    for Cifar datasets

    :param image: the image tensor

    :return: the preprocessed image
    """
    with tf_compat.name_scope("test_preprocess"):
        image = tf_compat.cast(image, dtype=tf_compat.float32)
        image = tf_compat_div(image, 255.0)
        image = tf_compat.image.random_crop(image, [32, 32, 3])
        return image


class CifarDataSet(ImageFolderDataset):
    """
    Base class for the Cifar datasets

    :param root: The root folder to find the dataset at,
        if not found will download here if download=True
    :param split: "train" for train split, or "test" for test split
    :param preprocess_fn: function to preprocess an image tensor,
        returning a new tensor
    :param download: True to download the images; False to reuse
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        image_size: int = 32,
        pre_resize_transforms: Union[SplitsTransforms, None] = SplitsTransforms(
            train=(preprocess_for_train,), val=(preprocess_for_eval,)
        ),
        post_resize_transforms: Union[SplitsTransforms, None] = SplitsTransforms(
            train=None,
            val=None,
        ),
        download: bool = True,
    ):
        create_dirs(root)
        self._download_dir = os.path.join(root, "download")
        self._extract_dir = os.path.join(root, "extract")
        self._train_dir = os.path.join(root, "train")
        self._test_dir = os.path.join(root, "test")
        if download and not os.path.exists(self._download_dir):
            self._download_and_extract()
            self._create_image_folders()

        self._per_pixel_mean = None
        super().__init__(
            root,
            train,
            image_size=image_size,
            pre_resize_transforms=pre_resize_transforms,
            post_resize_transforms=post_resize_transforms,
        )

    def __len__(self):
        """
        Number of images
        """
        if self._train:
            return 50000
        else:
            return 10000

    def name_scope(self) -> str:
        raise NotImplementedError()

    def _download_and_extract(self):
        raise NotImplementedError()

    def _create_image_folders(self):
        raise NotImplementedError()

    def _save_images(self, labels, data, filenames, image_dir):
        """
        Save images into corresponding label folder

        :param labels: list of labels of the images
        :param data: an numpy array of images, shape (num_images, 3027)
        :param filenames: list of file names of the images
        :param image_dir: directory to save images

        :return: the numpy array of images of shape (num_images, 32, 32, 3)
        """
        n_images = len(data)
        image_tensors = np.empty((n_images, _CIFAR_IMAGE_SIZE, _CIFAR_IMAGE_SIZE, 3))
        for idx, (label, data, filename) in tqdm(
            enumerate(zip(labels, data, filenames))
        ):
            img_array = data.reshape(
                (3, _CIFAR_IMAGE_SIZE, _CIFAR_IMAGE_SIZE)
            ).transpose((1, 2, 0))
            img = Image.fromarray(img_array)
            image_tensors[idx] = img
            img.save(os.path.join(image_dir, str(label), filename.decode("utf-8")))
        return image_tensors

    def per_pixel_mean(self):
        """
        Get the per pixel mean of the training images
        """
        return self._per_pixel_mean

    def _normalize_image(self, image: tf_compat.Tensor):
        """
        Normalize image

        :param image: the input image of shape (32, 32, 3)
        :return: the normalized image
        """
        return image - self.per_pixel_mean()


@DatasetRegistry.register(key=["cifar10"], attributes={"num_classes": 10})
class Cifar10DataSet(CifarDataSet):
    def __init__(
        self,
        root: str,
        train: bool = True,
        image_size: int = 32,
        pre_resize_transforms: Union[SplitsTransforms, None] = SplitsTransforms(
            train=(preprocess_for_train,), val=(preprocess_for_eval,)
        ),
        post_resize_transforms: Union[SplitsTransforms, None] = SplitsTransforms(
            train=None, val=None
        ),
        download: bool = True,
    ):
        super().__init__(
            root,
            train,
            image_size=image_size,
            pre_resize_transforms=pre_resize_transforms,
            post_resize_transforms=post_resize_transforms,
            download=download,
        )
        self._per_pixel_mean = np.load(
            os.path.join(self._train_dir, os.pardir, "per_pixel_mean_image.npy")
        )

    def name_scope(self) -> str:
        return "Cifar10"

    def _download_and_extract(self):
        """
        Download and extract the dataset into root
        """
        create_dirs(self._download_dir)
        file_path = os.path.join(self._download_dir, "cifar-10-python.tar.gz")
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        download_file(
            url,
            file_path,
            overwrite=False,
            progress_title="downloading CIFAR-10",
        )

        create_dirs(self._extract_dir)
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=self._extract_dir)

    def _create_image_folders(self):
        create_dirs(self._train_dir)
        create_dirs(self._test_dir)

        batches_dir = os.path.join(self._extract_dir, "cifar-10-batches-py")

        # Train
        image_tensors = []
        [create_dirs(os.path.join(self._train_dir, str(label))) for label in range(10)]
        batch_files = ["data_batch_{}".format(i) for i in range(1, 6)]
        for fname in batch_files:
            fpath = os.path.join(batches_dir, fname)
            print("Processing {}...".format(fpath))
            if not os.path.exists(fpath):
                raise ValueError("Train data batch {} not found".format(fpath))
            with open(fpath, "rb") as fo:
                batch_dict = pickle.load(fo, encoding="bytes")
            image_tensors.append(
                self._save_images(
                    batch_dict[b"labels"],
                    batch_dict[b"data"],
                    batch_dict[b"filenames"],
                    self._train_dir,
                )
            )
        image_tensors = np.concatenate(image_tensors)
        per_pixel_mean = np.mean(image_tensors, axis=0)
        np.save(
            os.path.join(self._train_dir, os.pardir, "per_pixel_mean_image.npy"),
            per_pixel_mean,
        )
        del image_tensors

        # Test
        [create_dirs(os.path.join(self._test_dir, str(label))) for label in range(10)]
        fpath = os.path.join(batches_dir, "test_batch")
        print("Processing {}...".format(fpath))
        if not os.path.exists(fpath):
            raise ValueError("Test data batch {} not found".format(fpath))
        with open(fpath, "rb") as fo:
            batch_dict = pickle.load(fo, encoding="bytes")
        self._save_images(
            batch_dict[b"labels"],
            batch_dict[b"data"],
            batch_dict[b"filenames"],
            self._test_dir,
        )


@DatasetRegistry.register(key=["cifar100"], attributes={"num_classes": 100})
class Cifar100DataSet(CifarDataSet):
    def __init__(
        self,
        root: str,
        train: bool = True,
        image_size: int = 32,
        pre_resize_transforms: Union[SplitsTransforms, None] = SplitsTransforms(
            train=(preprocess_for_train,), val=(preprocess_for_eval,)
        ),
        post_resize_transforms: Union[SplitsTransforms, None] = SplitsTransforms(
            train=None, val=None
        ),
        download: bool = True,
    ):
        super().__init__(
            root,
            train,
            image_size=image_size,
            pre_resize_transforms=pre_resize_transforms,
            post_resize_transforms=post_resize_transforms,
            download=download,
        )
        self._per_pixel_mean = np.load(
            os.path.join(self._train_dir, os.pardir, "per_pixel_mean_image.npy")
        )

    def name_scope(self) -> str:
        return "Cifar100"

    def _download_and_extract(self):
        """
        Download and extract the dataset into root
        """
        create_dirs(self._download_dir)
        file_path = os.path.join(self._download_dir, "cifar-100-python.tar.gz")
        url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        download_file(
            url,
            file_path,
            overwrite=False,
            progress_title="downloading CIFAR-100",
        )

        create_dirs(self._extract_dir)
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=self._extract_dir)

    def _create_image_folders(self):
        create_dirs(self._train_dir)
        create_dirs(self._test_dir)

        batches_dir = os.path.join(self._extract_dir, "cifar-100-python")

        # Train
        image_tensors = []
        [create_dirs(os.path.join(self._train_dir, str(label))) for label in range(100)]
        fpath = os.path.join(batches_dir, "train")
        print("Processing {}...".format(fpath))
        if not os.path.exists(fpath):
            raise ValueError("Train data batch {} not found".format(fpath))
        with open(fpath, "rb") as fo:
            batch_dict = pickle.load(fo, encoding="bytes")
        image_tensors.append(
            self._save_images(
                batch_dict[b"fine_labels"],
                batch_dict[b"data"],
                batch_dict[b"filenames"],
                self._train_dir,
            )
        )
        image_tensors = np.concatenate(image_tensors)
        per_pixel_mean = np.mean(image_tensors, axis=0)
        np.save(
            os.path.join(self._train_dir, os.pardir, "per_pixel_mean_image.npy"),
            per_pixel_mean,
        )
        del image_tensors

        # Test
        [create_dirs(os.path.join(self._test_dir, str(label))) for label in range(100)]
        fpath = os.path.join(batches_dir, "test")
        print("Processing {}...".format(fpath))
        if not os.path.exists(fpath):
            raise ValueError("Test data batch {} not found".format(fpath))
        with open(fpath, "rb") as fo:
            batch_dict = pickle.load(fo, encoding="bytes")
        self._save_images(
            batch_dict[b"fine_labels"],
            batch_dict[b"data"],
            batch_dict[b"filenames"],
            self._test_dir,
        )
