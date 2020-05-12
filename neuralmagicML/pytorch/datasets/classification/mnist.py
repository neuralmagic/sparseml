"""
MNIST dataset implementations for the image classification field in computer vision.
More info for the dataset can be found `here <http://yann.lecun.com/exdb/mnist/>`__.
"""

import torch
from torchvision.datasets import MNIST
from torchvision import transforms

from neuralmagicML.pytorch.datasets.registry import DatasetRegistry
from neuralmagicML.utils.datasets import default_dataset_path


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
