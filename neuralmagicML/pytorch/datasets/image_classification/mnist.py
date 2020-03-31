"""
MNIST dataset implementations for the image classification field in computer vision.
More info for the dataset can be found `here <http://yann.lecun.com/exdb/mnist/>`__.
"""

from torchvision.datasets import MNIST
from torchvision import transforms

from neuralmagicML.pytorch.datasets.registry import DatasetRegistry


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
    """

    def __init__(self, root: str, train: bool = True):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))]
        )
        super().__init__(root, train, transform, None, True)
