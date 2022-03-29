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

import functools
import logging
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch

from sparseml.pytorch.utils import default_device
from sparseml.utils.datasets import IMAGENET_RGB_MEANS, IMAGENET_RGB_STDS


try:
    import ffcv
    from ffcv import DatasetWriter, Loader
    from ffcv.fields import Field, IntField, RGBImageField
    from ffcv.fields.basics import IntDecoder
    from ffcv.fields.rgb_image import (
        CenterCropRGBImageDecoder,
        RandomResizedCropRGBImageDecoder,
    )
    from ffcv.loader import OrderOption
    from ffcv.pipeline.operation import Operation
    from ffcv.transforms import (
        NormalizeImage,
        RandomHorizontalFlip,
        Squeeze,
        ToDevice,
        ToTensor,
        ToTorchImage,
    )
except Exception as ffcv_error:
    ffcv = None
    ffcv_error = ffcv_error

LOGGER = logging.getLogger(__name__)


def timeit(func):
    """Decorator to time a function."""

    @functools.wraps(func)
    def time_closure(*args, **kwargs):
        """Function that actually does the timing."""

        start = time.perf_counter()
        result = func(*args, **kwargs)
        time_elapsed = time.perf_counter() - start
        LOGGER.info(f"Function: {func.__name__}, Time: {time_elapsed}")
        return result

    return time_closure


class FFCVCompatibleDataset(ABC):
    """
    Contract for adding FFCV functionality to a dataset.
    """

    @property
    @abstractmethod
    def validation(self):
        """
        :returns The dataset to use
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def ffcv_fields(self, *args, **kwargs) -> Dict[str, Field]:
        """
        :returns A dictionary of FFCV fields to use while writing the dataset.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def ffcv_pipelines(self, *args, **kwargs) -> Dict[str, List[Operation]]:
        """
        :returns: A Dict of operations to apply to the features and labels.
        """
        raise NotImplementedError()

    def _write(self, write_path: str, num_workers: int = 16) -> None:
        """
        Write the dataset to disk.
        Notes:
             - if the dataset is already written to disk, this will **NOT**
                overwrite the existing file.
             - This method must not be invoked directly. Use
                get_ffcv_loader(...) instead

        :pre-condition: The dataset has been initialized,
            and it's corresponding FFCV fields have been defined.
        """

        with self.switch_off_dataset_arg("transform"), self.switch_off_dataset_arg(
            "target_transform"
        ):
            self._write_if_not_written_already(
                write_path=write_path,
                num_workers=num_workers,
            )

        LOGGER.info("Dataset successfully written to disk.")

    @timeit
    def _write_if_not_written_already(
        self,
        write_path: str,
        num_workers: int = 16,
    ):
        dataset_path = Path(write_path)

        # Skip if dataset is already written
        if dataset_path.exists():
            LOGGER.info("Dataset already written to disk. Skipping _write.")
            return

        # Make parents if they don't exist
        dataset_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize the writer
        writer = DatasetWriter(
            fname=write_path,
            fields=self.ffcv_fields,
            num_workers=num_workers,
        )

        LOGGER.info(f"Writing dataset to {write_path}")

        # self is the dataset
        writer.from_indexed_dataset(self)

    @contextmanager
    def switch_off_dataset_arg(self, arg):
        """
        Context manager to perform an operation with a dataset argument turned
        to None
        """

        old_arg_value = None

        try:
            # Set dataset attribute to None
            if hasattr(self, arg):
                old_arg_value = getattr(self, arg)
                setattr(self, arg, None)
                LOGGER.info(f"Dataset attribute {arg} set to {None}")
            # Give back control
            yield

        finally:
            # Restore dataset attribute to its original state
            if old_arg_value is not None and hasattr(self, arg):
                setattr(self, arg, old_arg_value)
                LOGGER.info(f"Dataset attribute {arg} restored to old state")

    def get_ffcv_loader(
        self,
        write_path: str,
        num_workers: int = 16,
        batch_size: int = 32,
        distributed: int = 0,
        in_memory: int = 0,
        device: Union[str, int] = default_device(),
    ):
        """
        Initialize the data loader.

        :param write_path: The path to write the dataset to.
        :param num_workers: Number of workers to use for the data loader.
        :param batch_size: Batch size for the data loader.
        :param distributed: 1/0 Whether to use distributed data loading.
        :param in_memory: 1/0 Does the dataset fit in memory.
        """

        # Write the dataset if it hasn't been written already
        self._write(write_path=write_path, num_workers=num_workers)

        # Get the data loader
        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        if self.validation:
            order = OrderOption.SEQUENTIAL

        drop_last = not self.validation
        loader = Loader(
            fname=write_path,
            batch_size=batch_size,
            num_workers=num_workers,
            order=order,
            os_cache=in_memory,
            drop_last=drop_last,
            pipelines=self.ffcv_pipelines(device=device),
            distributed=distributed,
        )
        return loader


class FFCVImageNetDataset(FFCVCompatibleDataset):
    """
    A concrete implementation of the FFCVCompatibleDataset class for ImageNet.
    """

    IMAGENET_MEAN = np.array(IMAGENET_RGB_MEANS) * 255
    IMAGENET_STD = np.array(IMAGENET_RGB_STDS) * 255
    DEFAULT_CROP_RATIO = 224 / 256

    @property
    def ffcv_fields(self) -> Dict[str, Field]:
        """
        :returns: A dictionary of FFCV fields representing ImageNet
            based datasets.
        """
        return {
            "image": RGBImageField(),
            "label": IntField(),
        }

    @property
    def ffcv_pipelines(
        self,
        device: Union[str, int] = default_device(),
    ) -> Dict[str, List[Operation]]:
        """
        :returns: A Dict of operations to apply to the features and labels
            for ImageNet based datasets.
        """
        assert hasattr(self, "image_size"), "image_size not set"
        resolution = (self.image_size, self.image_size)

        decoder = (
            CenterCropRGBImageDecoder(
                resolution,
                ratio=self.DEFAULT_CROP_RATIO,
            )
            if self.validation
            else RandomResizedCropRGBImageDecoder(resolution)
        )
        image_pipeline: List[Operation] = [
            decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(torch.device(device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(self.IMAGENET_MEAN, self.IMAGENET_STD, np.float32),
        ]

        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(device), non_blocking=True),
        ]

        if self.validation:
            # remove randomness for validation step
            image_pipeline.pop(1)

        return {
            "image": image_pipeline,
            "label": label_pipeline,
        }

    @property
    def validation(self):
        """
        :returns: True if the dataset is used for validation.
        """
        # rand_trans are applied while training
        return not (hasattr(self, "rand_trans") and self.rand_trans)
