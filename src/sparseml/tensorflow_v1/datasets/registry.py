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
Code related to the TensorFlow dataset registry for easily creating datasets.
"""

from typing import Any, Dict, List, Union


__all__ = ["DatasetRegistry"]


class DatasetRegistry(object):
    """
    Registry class for creating datasets
    """

    _CONSTRUCTORS = {}
    _ATTRIBUTES = {}

    @staticmethod
    def create(key: str, *args, **kwargs):
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
