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
MNIST dataset implementations for the image classification field in computer vision.
More info for the dataset can be found `here <http://yann.lecun.com/exdb/mnist/>`__.
"""

import torch


try:
    from torchvision import transforms
    from torchvision.datasets import MNIST

    torchvision_import_error = None
except Exception as torchvision_error:
    MNIST = object  # default for constructor
    transforms = None
    torchvision_import_error = torchvision_error

from sparseml.pytorch.datasets.registry import DatasetRegistry
from sparseml.utils.datasets import default_dataset_path


__all__ = ["MNISTDataset"]


@DatasetRegistry.register(
    key=["mnist"],
    attributes={
        "num_classes": 10,
        "transform_means": [0.5],
        "transform_stds": [1.0],
        "num_channels": 1,
    },
)
class MNISTDataset(MNIST):
    """
    Wrapper for MNIST dataset to apply standard transforms.

    :param root: The root folder to find the dataset at, if not found will download here
    :param train: True if this is for the training distribution,
        False for the validation
    :param flatten: flatten the MNIST image from (1, 28, 28) to (784)
    """

    def __init__(
        self,
        root: str = default_dataset_path("mnist"),
        train: bool = True,
        flatten: bool = False,
    ):
        if torchvision_import_error is not None:
            raise torchvision_import_error

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))]
        )
        super().__init__(root, train, transform, None, True)
        self._flatten = flatten

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        if self._flatten:
            img = torch.flatten(img)

        return img, target
