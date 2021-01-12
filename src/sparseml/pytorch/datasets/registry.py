"""
Code related to the PyTorch dataset registry for easily creating datasets.
"""

from typing import Any, Dict, List, Union

from torch.utils.data import Dataset


__all__ = ["DatasetRegistry"]


class DatasetRegistry(object):
    """
    Registry class for creating datasets
    """

    _CONSTRUCTORS = {}
    _ATTRIBUTES = {}

    @staticmethod
    def create(key: str, *args, **kwargs) -> Dataset:
        """
        Create a new dataset for the given key

        :param key: the dataset key (name) to create
        :return: the instantiated model
        """
        if key not in DatasetRegistry._CONSTRUCTORS:
            raise ValueError(
                "key {} is not in the model registry; available: {}".format(
                    key, DatasetRegistry._CONSTRUCTORS.keys()
                )
            )

        return DatasetRegistry._CONSTRUCTORS[key](*args, **kwargs)

    @staticmethod
    def attributes(key: str) -> Dict[str, Any]:
        """
        :param key: the dataset key (name) to create
        :return: the specified attributes for the dataset
        """
        if key not in DatasetRegistry._CONSTRUCTORS:
            raise ValueError(
                "key {} is not in the model registry; available: {}".format(
                    key, DatasetRegistry._CONSTRUCTORS.keys()
                )
            )

        return DatasetRegistry._ATTRIBUTES[key]

    @staticmethod
    def register(key: Union[str, List[str]], attributes: Dict[str, Any]):
        """
        Register a dataset with the registry. Should be used as a decorator

        :param key: the model key (name) to create
        :param attributes: the specified attributes for the dataset
        :return: the decorator
        """
        if not isinstance(key, List):
            key = [key]

        def decorator(const_func):
            for r_key in key:
                if r_key in DatasetRegistry._CONSTRUCTORS:
                    raise ValueError("key {} is already registered".format(key))

                DatasetRegistry._CONSTRUCTORS[r_key] = const_func
                DatasetRegistry._ATTRIBUTES[r_key] = attributes

            return const_func

        return decorator
