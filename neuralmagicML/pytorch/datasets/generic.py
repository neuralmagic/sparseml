from typing import Union, Tuple
import re
import torch
from torch.utils.data import Dataset
from torchvision import transforms


__all__ = ["EarlyStopDataset", "NoisyDataset", "RandNDataset", "CacheableDataset"]


class EarlyStopDataset(Dataset):
    def __init__(self, original: Dataset, early_stop: int):
        """
        Dataset that handles applying an early stop when iterating through the dataset
        IE will allow indexing between [0, early_stop)

        :param original: the original dataset to apply an early stop to
        :param early_stop: the total number of data items to run through, if -1 then will go through whole dataset
        """
        self._original = original
        self._early_stop = early_stop

        if self._early_stop > len(self._original):
            raise ValueError(
                "Cannot apply early stop of {}, its greater than length of dataset {}".format(
                    self._early_stop, len(self._original)
                )
            )

    def __getitem__(self, index):
        return self._original.__getitem__(index)

    def __len__(self):
        return self._early_stop if self._early_stop > 0 else self._original.__len__()

    def __repr__(self):
        rep = self._original.__str__()
        rep = re.sub(
            r"Number of datapoints:[ ]+[0-9]+",
            "Number of datapoints: {}".format(self.__len__()),
            rep,
        )

        return rep


class NoisyDataset(Dataset):
    def __init__(self, original: Dataset, intensity: float):
        """
        Add random noise from a standard distribution mean(0) and stdev(intensity) on top of a dataset

        :param original: the dataset to add noise on top of
        :param intensity: the level of noise to add (creates the noise with this standard deviation)
        """
        self._original = original
        self._intensity = intensity

    def __getitem__(self, index):
        x_tens, y_tens = self._original.__getitem__(index)
        noise = torch.zeros(x_tens.size()).normal_(mean=0, std=self._intensity)
        x_tens += noise

        return x_tens, y_tens

    def __len__(self):
        return self._original.__len__()


class RandNDataset(Dataset):
    def __init__(
        self, length: int, shape: Union[int, Tuple[int, ...]], normalize: bool
    ):
        """
        Generates a random dataset

        :param length: the number of random items to create in the dataset
        :param shape: the shape of the data to create
        :param normalize: Normalize the data according to imagenet distribution (must be of shape 3,x,x)
        """
        if isinstance(shape, int):
            shape = (3, shape, shape)

        self._data = []
        normalize = (
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if normalize
            else None
        )

        for _ in range(length):
            tens = torch.randn(*shape)

            if normalize:
                tens = normalize(tens)

            self._data.append(tens)

    def __getitem__(self, index):
        return self._data[index], torch.tensor(1)

    def __len__(self):
        return len(self._data)


class CacheableDataset(Dataset):
    def __init__(self, original: Dataset):
        """
        Generates a cacheable dataset, ie stores the data in a cache in cpu memory
        so it doesn't have to be loaded from disk every time

        Note, this can only be used with a data loader that has num_workers=0

        :param original: the original dataset to cache
        """
        self._original = original
        self._cache = {}

    def __getitem__(self, index):
        if index not in self._cache:
            self._cache[index] = self._original[index]

        return self._cache[index]

    def __len__(self):
        return self._original.__len__()
