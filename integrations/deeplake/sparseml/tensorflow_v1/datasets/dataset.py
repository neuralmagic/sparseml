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
General dataset implementations for TensorFlow
"""

from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Tuple

from sparseml.tensorflow_v1.utils import tf_compat


__all__ = [
    "create_split_iterators_handle",
    "Dataset",
]


def _make_initializable_iterator(dataset: tf_compat.data.Dataset):
    """
    Make initializable iterator with different versions of TF

    :param dataset: the dataset to create the iterator
    :return: an iterator
    """
    if hasattr(tf_compat.data, "make_initializable_iterator"):
        return tf_compat.data.make_initializable_iterator(dataset)
    else:
        return dataset.make_initializable_iterator()


def create_split_iterators_handle(split_datasets: Iterable) -> Tuple[Any, Any, List]:
    """
    Create an iterators handle for switching between datasets easily while training.

    :param split_datasets: the datasets to create the splits and handle for
    :return: a tuple containing the handle that should be set with a feed dict,
        the iterator used to get the next batch,
        and a list of the iterators created from the split_datasets
    """
    output_types = None
    output_shapes = None
    split_iterators = []

    for split_dataset in split_datasets:
        # get_output_types and shapes are not available in TF 1.13 and prior
        # hence the following conditional assignments
        output_types = (
            tf_compat.data.get_output_types(split_dataset)
            if hasattr(tf_compat.data, "get_output_types")
            else split_dataset.output_types
        )
        output_shapes = (
            tf_compat.data.get_output_shapes(split_dataset)
            if hasattr(tf_compat.data, "get_output_shapes")
            else split_dataset.output_shapes
        )
        split_iterators.append(_make_initializable_iterator(split_dataset))

    handle = tf_compat.placeholder(tf_compat.string, shape=[])
    iterator = tf_compat.data.Iterator.from_string_handle(
        handle, output_types, output_shapes
    )

    return handle, iterator, split_iterators


class Dataset(metaclass=ABCMeta):
    """
    Generic dataset implementation for TensorFlow.
    Expected to work with the tf.data APIs
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
    ) -> tf_compat.data.Dataset:
        """
        Create the dataset in the current graph using tf.data APIs

        :param batch_size: the batch size to create the dataset for
        :param repeat_count: the number of times to repeat the dataset,
            if unset or None, will repeat indefinitely
        :param shuffle_buffer_size: None if not shuffling,
            otherwise the size of the buffer to use for shuffling data
        :param prefetch_buffer_size: None if not prefetching,
            otherwise the size of the buffer to use for buffering
        :param num_parallel_calls: the number of parallel calls to run the
            processor function with
        :return: a tf.data.Dataset instance
        """
        with tf_compat.name_scope(self.name_scope()):
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

    def build_input_fn(
        self,
        batch_size: int,
        repeat_count: int = None,
        shuffle_buffer_size: int = None,
        prefetch_buffer_size: int = None,
        num_parallel_calls: int = None,
    ) -> Callable[[], Tuple[Dict[str, tf_compat.Tensor], Dict[str, tf_compat.Tensor]]]:
        """
        Create an input_fn to be used with Estimators.
        Invocation of the input_fn will create the dataset in the current graph
        as well as return a tuple containing
        (a dictionary of feature tensors, a dictionary of label tensors).

        :param batch_size: the batch size to create the dataset for
        :param repeat_count: the number of times to repeat the dataset,
            if unset or None, will repeat indefinitely
        :param shuffle_buffer_size: None if not shuffling,
            otherwise the size of the buffer to use for shuffling data
        :param prefetch_buffer_size: None if not prefetching,
            otherwise the size of the buffer to use for buffering
        :param num_parallel_calls: the number of parallel calls to run the
            processor function with
        :return: a callable representing the input_fn for an Estimator
        """

        def input_fn() -> Tuple[
            Dict[str, tf_compat.Tensor], Dict[str, tf_compat.Tensor]
        ]:
            dataset = self.build(
                batch_size,
                repeat_count,
                shuffle_buffer_size,
                prefetch_buffer_size,
                num_parallel_calls,
            )
            dataset_iter = _make_initializable_iterator(dataset)
            tf_compat.add_to_collection(
                tf_compat.GraphKeys.TABLE_INITIALIZERS, dataset_iter.initializer
            )
            iter_batch = dataset_iter.get_next()
            features, labels = self.format_iterator_batch(iter_batch)

            return features, labels

        return input_fn

    @abstractmethod
    def creator(self) -> tf_compat.data.Dataset:
        """
        Implemented by sub classes to create a tf.data dataset for the given impl.

        :return: a created tf.data dataset
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

    @abstractmethod
    def format_iterator_batch(
        self, iter_batch: Tuple[tf_compat.Tensor, ...]
    ) -> Tuple[Dict[str, tf_compat.Tensor], Dict[str, tf_compat.Tensor]]:
        """
        Implemented by sub classes to parse the output from make_one_shot_iterator
        into a features and labels dict to be used with Estimators

        :param iter_batch: the batch ref returned from the iterator
        :return: a tuple containing
            (a dictionary of feature tensors, a dictionary of label tensors)
        """
        raise NotImplementedError()

    @abstractmethod
    def name_scope(self) -> str:
        """
        Implemented by sub classes to get a name scope for building the dataset
        in the graph

        :return: the name scope the dataset should be built under in the graph
        """
        raise NotImplementedError()
