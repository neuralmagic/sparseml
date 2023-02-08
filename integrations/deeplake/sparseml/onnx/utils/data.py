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
Utilities for data loading into numpy for use in ONNX supported systems
"""

import logging
import math
from collections import OrderedDict
from typing import Dict, List, Tuple, Union

import numpy
from onnx import ModelProto

from sparseml.onnx.utils.helpers import (
    check_load_model,
    extract_shape,
    get_numpy_dtype,
    model_inputs,
    model_outputs,
)
from sparseml.utils import NumpyArrayBatcher, load_labeled_data


__all__ = ["DataLoader"]


_LOGGER = logging.getLogger(__name__)


class DataLoader(object):
    """
    Data loader instance that supports loading numpy arrays from file or memory
    and creating an iterator to go through batches of that data.

    Iterator returns a tuple containing (data, label).
    label is only returned if label data was passed in.

    :param data: a file glob pointing to numpy files, path to a tar ball of numpy
        files, or loaded numpy data
    :param labels: a file glob pointing to numpy files path to a tar ball of numpy
        files, or loaded numpy data
    :param batch_size: the size of batches to create for the iterator
    :param iter_steps: the number of steps (batches) to create.
        Set to -1 for infinite, 0 for running through the loaded data once,
        or a positive integer for the desired number of steps
    """

    @staticmethod
    def from_random(
        data_shapes: Dict[str, Tuple[int, ...]],
        label_shapes: Union[None, Dict[str, Tuple[int, ...]]],
        batch_size: int,
        iter_steps: int = 0,
        num_samples: int = 100,
        data_types: Dict[str, numpy.dtype] = None,
    ):
        """
        Create a DataLoader from random data

        :param data_shapes: shapes to create for the data items
        :param label_shapes: shapes to create for the label items
        :param batch_size: the size of batches to create for the iterator
        :param iter_steps: the number of steps (batches) to create.
            Set to -1 for infinite, 0 for running through the loaded data once,
            or a positive integer for the desired number of steps
        :param num_samples: number of random samples to create
        :param data_types: optional numpy data types for each of the data items
        :return: the created DataLoader instance with the random data
        """
        # Validate that shapes are integers and positive
        for shape in data_shapes.values():
            if not all(isinstance(dim, int) and dim > 0 for dim in shape):
                raise RuntimeError(
                    "Invalid input shape, cannot create a random input shape"
                    " from: {}".format(shape)
                )
        data = []
        for _ in range(num_samples):
            batch = OrderedDict()
            for key in data_shapes:
                # generate a random array based on the supported data type
                dtype = (
                    data_types[key]
                    if data_types is not None and key in data_types
                    else numpy.float32
                )
                dtype_name = dtype.name if hasattr(dtype, "name") else dtype.__name__
                if dtype is not None and "float" in dtype_name:
                    array = numpy.random.random(data_shapes[key]).astype(dtype)
                elif dtype is not None and "int" in dtype_name:
                    iinfo = numpy.iinfo(dtype)
                    array = numpy.random.randint(
                        iinfo.min, iinfo.max, data_shapes[key], dtype
                    )
                elif dtype is not None and dtype is numpy.bool:
                    array = numpy.random.random(data_shapes[key]) < 0.5
                else:
                    raise RuntimeError(
                        "Cannot create random input for"
                        " {} with unsupported type {}".format(key, dtype)
                    )
                batch[key] = array
            data.append(batch)
        _LOGGER.debug(
            "created random data of shapes {} and len {}".format(data_shapes, len(data))
        )
        labels = (
            [
                OrderedDict(
                    [
                        (
                            key,
                            numpy.ascontiguousarray(
                                numpy.random.random(shape).astype(numpy.float32)
                            ),
                        )
                        for key, shape in label_shapes.items()
                    ]
                )
                for _ in range(num_samples)
            ]
            if label_shapes is not None
            else None
        )

        if labels:
            _LOGGER.debug(
                "created random labels of shapes {} and len {}".format(
                    data_shapes, len(data)
                )
            )
        else:
            _LOGGER.debug("skipping creation of labels")

        return DataLoader(data, labels, batch_size, iter_steps)

    @staticmethod
    def from_model_random(
        model: Union[str, ModelProto],
        batch_size: int,
        iter_steps: int = 0,
        num_samples: int = 100,
        create_labels: bool = False,
        strip_first_dim: bool = True,
    ):
        """
        Create a DataLoader from random data for a model's input and output sizes

        :param model: the loaded model or a file path to the onnx model
            to create random data for
        :param batch_size: the size of batches to create for the iterator
        :param iter_steps: the number of steps (batches) to create.
            Set to -1 for infinite, 0 for running through the loaded data once,
            or a positive integer for the desired number of steps
        :param num_samples: number of random samples to create
        :param create_labels: True to create random label data as well, False otherwise
        :param strip_first_dim: True to strip the first dimension from the inputs
            and outputs, typically the batch dimension
        :return: the created DataLoader instance with the random data
        """
        model = check_load_model(model)
        inputs = model_inputs(model)
        outputs = model_outputs(model)
        data_shapes = OrderedDict(
            [
                (
                    inp.name,
                    extract_shape(inp)[1:] if strip_first_dim else extract_shape(inp),
                )
                for inp in inputs
            ]
        )
        data_types = OrderedDict([(inp.name, get_numpy_dtype(inp)) for inp in inputs])
        _LOGGER.debug("pulled input shapes {} from the model".format(data_shapes))
        label_shapes = (
            OrderedDict(
                [
                    (
                        out.name,
                        extract_shape(out)[1:]
                        if strip_first_dim
                        else extract_shape(out),
                    )
                    for out in outputs
                ]
            )
            if create_labels
            else None
        )

        if label_shapes:
            _LOGGER.debug(
                "pulled label output shapes {} from the model".format(data_shapes)
            )
        else:
            _LOGGER.debug("skipping pulling label shapes")

        return DataLoader.from_random(
            data_shapes, label_shapes, batch_size, iter_steps, num_samples, data_types
        )

    def __init__(
        self,
        data: Union[str, List[Dict[str, numpy.ndarray]]],
        labels: Union[None, str, List[Union[numpy.ndarray, Dict[str, numpy.ndarray]]]],
        batch_size: int,
        iter_steps: int = 0,
    ):
        self._batch_size = batch_size
        self._iter_steps = iter_steps
        self._labeled_data = load_labeled_data(data, labels, raise_on_error=False)

        if len(self._labeled_data) < 1:
            raise ValueError(
                "No data for DataLoader after loading. data: {}, labels: {}".format(
                    data, labels
                )
            )

        self._index = 0
        self._step_count = 0

        if self.infinite:
            # __len__ cannot return math.inf as a value and must be non-negative integer
            self._max_steps = 0
        elif self._iter_steps > 0:
            self._max_steps = self._iter_steps
        else:
            self._max_steps = math.ceil(
                len(self._labeled_data) / float(self._batch_size)
            )

    @property
    def batch_size(self) -> int:
        """
        :return: the size of batches to create for the iterator
        """
        return self._batch_size

    @property
    def iter_steps(self) -> int:
        """
        :return: the number of steps (batches) to create.
            Set to -1 for infinite, 0 for running through the loaded data once,
            or a positive integer for the desired number of steps
        """
        return self._iter_steps

    @property
    def labeled_data(
        self,
    ) -> List[
        Tuple[
            Union[numpy.ndarray, Dict[str, numpy.ndarray]],
            Union[None, numpy.ndarray, Dict[str, numpy.ndarray]],
        ]
    ]:
        """
        :return: the loaded data and labels
        """
        return self._labeled_data

    @property
    def infinite(self) -> bool:
        """
        :return: True if the loader instance is setup to continually create batches,
            False otherwise
        """
        return self._iter_steps == -1

    def __len__(self):
        return self._max_steps

    def __iter__(self):
        self._index = 0
        self._step_count = 0

        return self

    def __next__(
        self,
    ) -> Tuple[Dict[str, numpy.ndarray], Union[None, Dict[str, numpy.ndarray]]]:
        if not self.infinite and self._step_count >= self._max_steps:
            _LOGGER.debug("reached in of dataset, raising StopIteration")
            raise StopIteration()

        self._step_count += 1
        data_batcher = NumpyArrayBatcher()
        label_batcher = NumpyArrayBatcher()
        num_resets = 0

        while len(data_batcher) < self._batch_size:
            try:
                _LOGGER.debug("including data in batch at index {}".format(self._index))
                dat, lab = self._labeled_data[self._index]

                if lab is None and len(label_batcher) > 0:
                    raise ValueError(
                        (
                            "data has no label at index {}, but other data had labels"
                        ).format(self._index)
                    )
                elif (
                    lab is not None
                    and len(label_batcher) == 0
                    and len(data_batcher) > 0
                ):
                    raise ValueError(
                        (
                            "data has label at index {}, "
                            "but other data did not have labels"
                        ).format(self._index)
                    )
                elif lab is not None:
                    label_batcher.append(lab)

                data_batcher.append(dat)
            except Exception as err:
                logging.error(
                    (
                        "DataLoader: Error while adding file "
                        "to batch for index {}: {}"
                    ).format(self._index, err)
                )

            if self._index >= len(self._labeled_data) - 1:
                _LOGGER.debug("resetting index to loop data again")
                self._index = 0
                num_resets += 1

                if num_resets > self._batch_size // len(self._labeled_data) + 2:
                    # make sure we're not in an infinite loop because none of the
                    # data was loadable
                    raise ValueError(
                        "could not create a batch from the files, "
                        "not enough were loadable to fill the batch size"
                    )
            else:
                self._index += 1

        batch_data = data_batcher.stack()
        _LOGGER.debug("created batch data of size {}".format(len(batch_data)))
        batch_label = label_batcher.stack() if len(label_batcher) > 0 else None

        if batch_label:
            _LOGGER.debug("created batch labels of size {}".format(len(batch_label)))

        return batch_data, batch_label
