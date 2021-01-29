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
from sparseml.tensorflow_v1.datasets.classification.imagefolder import (
    ImageFolderDataset,
)
from sparseml.tensorflow_v1.datasets.registry import DatasetRegistry
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
class ImageNetDataset(ImageFolderDataset):
    """
    ImageNet dataset implementation

    :param root: the root location for the dataset's images to load
    :param train: True to load the training dataset from the root,
        False for validation
    :param image_size: the size of the image to reshape to
    """

    def __init__(
        self,
        root: str = default_dataset_path("imagenet"),
        train: bool = True,
        image_size: int = 224,
    ):
        super().__init__(root, train, image_size)

    def name_scope(self) -> str:
        """
        :return: the name scope the dataset should be built under in the graph
        """
        return "Imagenet_{}".format("train" if self._train else "val")
