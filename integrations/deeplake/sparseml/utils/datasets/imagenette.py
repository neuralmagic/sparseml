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
General utilities for the imagenette and imagewoof dataset implementations
for the image classification field in computer vision.
More info for the dataset can be found `here <https://github.com/fastai/imagenette>`__.
"""

import os
import tarfile
from enum import Enum

from sparseml.utils import clean_path, create_dirs
from sparsezoo.utils import download_file


__all__ = [
    "ImagenetteSize",
    "ImagenetteDownloader",
    "ImagewoofDownloader",
    "IMAGENETTE_CLASSES",
]


IMAGENETTE_CLASSES = {
    0: "tench",
    1: "English springer",
    2: "cassette player",
    3: "chain saw",
    4: "church",
    5: "French horn",
    6: "garbage truck",
    7: "gas pump",
    8: "golf ball",
    9: "parachute",
}


class ImagenetteSize(Enum):
    """
    Dataset size for Imagenette / Imagewoof.
    full does not resize the original dataset at all.
    s320 resizes the images to 320px.
    s160 resizes the images to 160px.
    """

    full = "full"
    s320 = "s320"
    s160 = "s160"


class ImagenetteDownloader(object):
    """
    Downloader implementation for the imagenette dataset.
    More info on the dataset can be found
    `here <https://github.com/fastai/imagenette>`__

    :param download_root: the local path to download the files to
    :param dataset_size: which dataset size to download
    :param download: True to run the download, False otherwise.
        If False, dataset must already exist at root.
    """

    def __init__(
        self, download_root: str, dataset_size: ImagenetteSize, download: bool
    ):
        self._download_root = clean_path(download_root)
        self._dataset_size = dataset_size
        self._download = download

        if dataset_size == ImagenetteSize.s160:
            self._extract_name = "imagenette-160"
        elif dataset_size == ImagenetteSize.s320:
            self._extract_name = "imagenette-320"
        elif dataset_size == ImagenetteSize.full:
            self._extract_name = "imagenette"
        else:
            raise ValueError("Unknown ImagenetteSize given of {}".format(dataset_size))

        self._extracted_root = os.path.join(self._download_root, self._extract_name)

        if download:
            self._download_and_extract()
        else:
            file_path = "{}.tar".format(self._extracted_root)

            if not os.path.exists(file_path):
                raise ValueError(
                    "could not find original tar for the dataset at {}".format(
                        file_path
                    )
                )

    @property
    def download_root(self) -> str:
        """
        :return: the local path to download the files to
        """
        return self._download_root

    @property
    def dataset_size(self) -> ImagenetteSize:
        """
        :return: which dataset size to download
        """
        return self._dataset_size

    @property
    def download(self) -> bool:
        """
        :return: True to run the download, False otherwise.
            If False, dataset must already exist at root.
        """
        return self._download

    @property
    def extracted_root(self) -> str:
        """
        :return: Where the specific dataset was extracted to
        """
        return self._extracted_root

    def split_root(self, train: bool) -> str:
        """
        :param train: True to get the path to the train dataset, False for validation
        :return: The path to the desired split for the dataset
        """
        return os.path.join(self.extracted_root, "train" if train else "val")

    def _download_and_extract(self):
        if self._dataset_size == ImagenetteSize.full:
            url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette.tgz"
        elif self._dataset_size == ImagenetteSize.s320:
            url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette-320.tgz"
        elif self._dataset_size == ImagenetteSize.s160:
            url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette-160.tgz"
        else:
            raise ValueError(
                "unknown imagenette size given of {}".format(self._dataset_size)
            )

        create_dirs(self._extracted_root)
        file_path = "{}.tar".format(self._extracted_root)

        if os.path.exists(file_path):
            print("already downloaded imagenette {}".format(self._dataset_size))

            return

        download_file(
            url,
            file_path,
            overwrite=False,
            progress_title="downloading imagenette {}".format(self._dataset_size),
        )

        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=self.download_root)


class ImagewoofDownloader(object):
    """
    Downloader implementation for the imagewoof dataset.
    More info on the dataset can be found
    `here <https://github.com/fastai/imagenette>`__

    :param root: the local path to download the files to
    :param dataset_size: which dataset size to download
    :param download: True to run the download, False otherwise.
        If False, dataset must already exist at root.
    """

    def __init__(
        self, download_root: str, dataset_size: ImagenetteSize, download: bool
    ):
        self._download_root = clean_path(download_root)
        self._dataset_size = dataset_size
        self._download = download

        if dataset_size == ImagenetteSize.s160:
            self._extract_name = "imagewoof-160"
        elif dataset_size == ImagenetteSize.s320:
            self._extract_name = "imagewoof-320"
        elif dataset_size == ImagenetteSize.full:
            self._extract_name = "imagewooof"
        else:
            raise ValueError("Unknown ImagenetteSize given of {}".format(dataset_size))

        self._extracted_root = os.path.join(self._download_root, self._extract_name)

        if download:
            self._download_and_extract()
        else:
            file_path = "{}.tar".format(self._extracted_root)

            if not os.path.exists(file_path):
                raise ValueError(
                    "could not find original tar for the dataset at {}".format(
                        file_path
                    )
                )

    @property
    def download_root(self) -> str:
        """
        :return: the local path to download the files to
        """
        return self._download_root

    @property
    def dataset_size(self) -> ImagenetteSize:
        """
        :return: which dataset size to download
        """
        return self._dataset_size

    @property
    def download(self) -> bool:
        """
        :return: True to run the download, False otherwise.
            If False, dataset must already exist at root.
        """
        return self._download

    @property
    def extracted_root(self) -> str:
        """
        :return: Where the specific dataset was extracted to
        """
        return self._extracted_root

    def split_root(self, train: bool) -> str:
        """
        :param train: True to get the path to the train dataset, False for validation
        :return: The path to the desired split for the dataset
        """
        return os.path.join(self.extracted_root, "train" if train else "val")

    def _download_and_extract(self):
        if self._dataset_size == ImagenetteSize.full:
            url = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof.tgz"
        elif self._dataset_size == ImagenetteSize.s320:
            url = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof-320.tgz"
        elif self._dataset_size == ImagenetteSize.s160:
            url = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof-160.tgz"
        else:
            raise ValueError(
                "unknown imagenette size given of {}".format(self._dataset_size)
            )

        create_dirs(self._extracted_root)
        file_path = "{}.tar".format(self._extracted_root)

        if os.path.exists(file_path):
            print("already downloaded imagewoof {}".format(self._dataset_size))

            return

        download_file(
            url,
            file_path,
            overwrite=False,
            progress_title="downloading imagewoof {}".format(self._dataset_size),
        )

        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=self.download_root)
