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
General dataset implementations for Keras
"""

from abc import ABCMeta, abstractmethod

import tensorflow


__all__ = [
    "Dataset",
]


class Dataset(metaclass=ABCMeta):
    """
    Generic dataset implementation for Keras.
    Expected to work with the tensorflow.data APIs
    """

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    def build(
        self,
        batch_size: int,
        repeat_count: int = None,
        shuffle_buffer_size: int = None,
        prefetch_buffer_size: int = None,
        num_parallel_calls: int = None,
    ) -> tensorflow.data.Dataset:
        """
        Create the dataset in the current graph using tensorflow.data APIs
        :param batch_size: the batch size to create the dataset for
        :param repeat_count: the number of times to repeat the dataset,
            if unset or None, will repeat indefinitely
        :param shuffle_buffer_size: None if not shuffling,
            otherwise the size of the buffer to use for shuffling data
        :param prefetch_buffer_size: None if not prefetching,
            otherwise the size of the buffer to use for buffering
        :param num_parallel_calls: the number of parallel calls to run the
            processor function with
        :return: a tensorflow.data.Dataset instance
        """
        dataset = self.creator()

        if shuffle_buffer_size and shuffle_buffer_size > 0:
            dataset = dataset.shuffle(
                shuffle_buffer_size, reshuffle_each_iteration=True
            )

        dataset = dataset.map(self.processor, num_parallel_calls=num_parallel_calls)

        # Together with shuffling above, putting batch after repeat yields
        # batches that straddle epoch boundaries
        dataset = dataset.repeat(repeat_count)
        dataset = dataset.batch(batch_size)

        if prefetch_buffer_size and prefetch_buffer_size > 0:
            dataset = dataset.prefetch(prefetch_buffer_size)

        return dataset

    @abstractmethod
    def creator(self) -> tensorflow.data.Dataset:
        """
        Implemented by sub classes to create a tensorflow.data dataset for the given
        impl.

        :return: a created tensorflow.data dataset
        """
        raise NotImplementedError()

    @abstractmethod
    def processor(self, *args, **kwargs):
        """
        Implemented by sub classes to parallelize and map processing functions
        for loading the data of the dataset into memory.
        :param args: generic inputs for processing
        :param kwargs: generic inputs for processing
        :return: the processed tensors
        """
        raise NotImplementedError()
