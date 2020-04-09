from torch.utils.data import Dataset

from neuralmagicML.pytorch.datasets import (
    DatasetRegistry,
    CIFAR10Dataset,
    CIFAR100Dataset,
)


def _validate_cifar(dataset: Dataset, num_classes: int):
    assert len(dataset) > 0
    item = dataset[0]
    assert isinstance(item, tuple)
    assert item[0].shape[0] == 3
    assert item[0].shape[1] == 32
    assert item[0].shape[2] == 32
    assert item[1] < num_classes


def test_cifar_10():
    train_dataset = CIFAR10Dataset(train=True)
    _validate_cifar(train_dataset, 10)

    val_dataset = CIFAR10Dataset(train=False)
    _validate_cifar(val_dataset, 10)

    reg_dataset = DatasetRegistry.create("cifar10", train=False)
    _validate_cifar(reg_dataset, 10)


def test_cifar_100():
    train_dataset = CIFAR100Dataset(train=True)
    _validate_cifar(train_dataset, 100)

    val_dataset = CIFAR100Dataset(train=False)
    _validate_cifar(val_dataset, 100)

    reg_dataset = DatasetRegistry.create("cifar100", train=False)
    _validate_cifar(reg_dataset, 100)
