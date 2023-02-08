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
CIFAR dataset implementations for the image classification field in computer vision.
More info for the dataset can be found
`here <https://www.cs.toronto.edu/~kriz/cifar.html>`__.
"""

try:
    from torchvision import transforms
    from torchvision.datasets import CIFAR10, CIFAR100

    torchvision_import_error = None
except Exception as torchvision_error:
    CIFAR10 = object  # default for constructor
    CIFAR100 = object  # default for constructor
    transforms = None
    torchvision_import_error = torchvision_error

from sparseml.pytorch.datasets.registry import DatasetRegistry
from sparseml.utils.datasets import default_dataset_path


__all__ = ["CIFAR10Dataset", "CIFAR100Dataset"]


_CIFAR10_RGB_MEANS = [0.491, 0.482, 0.447]
_CIFAR10_RGB_STDS = [0.247, 0.243, 0.262]


@DatasetRegistry.register(
    key=["cifar10", "cifar_10"],
    attributes={
        "num_classes": 10,
        "transform_means": _CIFAR10_RGB_MEANS,
        "transform_stds": _CIFAR10_RGB_STDS,
    },
)
class CIFAR10Dataset(CIFAR10):
    """
    Wrapper for the CIFAR10 dataset to apply standard transforms.

    :param root: The root folder to find the dataset at, if not found will download here
    :param train: True if this is for the training distribution, false for the
        validation
    :param rand_trans: True to apply RandomCrop and RandomHorizontalFlip to the data,
        False otherwise
    """

    def __init__(
        self,
        root: str = default_dataset_path("cifar10"),
        train: bool = True,
        rand_trans: bool = False,
    ):
        if torchvision_import_error is not None:
            raise torchvision_import_error

        if rand_trans:
            trans = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            trans = []

        trans.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=_CIFAR10_RGB_MEANS, std=_CIFAR10_RGB_STDS),
            ]
        )

        super().__init__(root, train, transforms.Compose(trans), None, True)


_CIFAR100_RGB_MEANS = [0.507, 0.487, 0.441]
_CIFAR100_RGB_STDS = [0.267, 0.256, 0.276]


@DatasetRegistry.register(
    key=["cifar100", "cifar_100"],
    attributes={
        "num_classes": 100,
        "transform_means": _CIFAR100_RGB_MEANS,
        "transform_stds": _CIFAR100_RGB_STDS,
    },
)
class CIFAR100Dataset(CIFAR100):
    """
    Wrapper for the CIFAR100 dataset to apply standard transforms.

    :param root: The root folder to find the dataset at, if not found will download here
    :param train: True if this is for the training distribution,
        False for the validation
    :param rand_trans: True to apply RandomCrop and RandomHorizontalFlip to the data,
        False otherwise
    """

    def __init__(
        self,
        root: str = default_dataset_path("cifar100"),
        train: bool = True,
        rand_trans: bool = False,
    ):
        normalize = transforms.Normalize(
            mean=_CIFAR100_RGB_MEANS, std=_CIFAR100_RGB_STDS
        )
        trans = (
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
            if rand_trans
            else []
        )
        trans.extend([transforms.ToTensor(), normalize])
        super().__init__(root, train, transforms.Compose(trans), None, True)
