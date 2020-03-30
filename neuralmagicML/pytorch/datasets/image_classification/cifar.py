from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms

from neuralmagicML.pytorch.datasets.utils import DATASET_MAPPINGS


__all__ = ["CIFAR10Dataset", "CIFAR100Dataset"]


class CIFAR10Dataset(CIFAR10):
    """
    Wrapper for the CIFAR10 dataset to apply standard transforms
    """

    def __init__(self, root: str, train: bool = True, rand_trans: bool = False):
        """
        :param root: The root folder to find the dataset at, if not found will download here
        :param train: True if this is for the training distribution, false for the validation
        :param rand_trans: True to apply RandomCrop and RandomHorizontalFlip to the data, False otherwise
        """
        if rand_trans:
            trans = [
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            trans = [transforms.Resize(32), transforms.CenterCrop(32)]

        trans.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
                ),
            ]
        )

        super().__init__(root, train, transforms.Compose(trans), None, True)


DATASET_MAPPINGS["cifar10"] = CIFAR10Dataset, 10


class CIFAR100Dataset(CIFAR100):
    """
    Wrapper for the CIFAR100 dataset to apply standard transforms
    """

    def __init__(self, root: str, train: bool = True, rand_trans: bool = False):
        """
        :param root: The root folder to find the dataset at, if not found will download here
        :param train: True if this is for the training distribution, false for the validation
        :param rand_trans: True to apply RandomCrop and RandomHorizontalFlip to the data, False otherwise
        """
        normalize = transforms.Normalize(
            mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]
        )
        trans = (
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
            if rand_trans
            else []
        )
        trans.extend([transforms.ToTensor(), normalize])
        super().__init__(root, train, transforms.Compose(trans), None, True)


DATASET_MAPPINGS["cifar100"] = CIFAR100Dataset, 100
