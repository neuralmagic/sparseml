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

import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch

from ffcv.fields.basics import IntDecoder
from ffcv.loader import OrderOption
from sparseml.pytorch.utils import default_device


try:
    import ffcv
    from ffcv import DatasetWriter, Loader
    from ffcv.fields import IntField, RGBImageField
    from ffcv.fields.rgb_image import (
        CenterCropRGBImageDecoder,
        RandomResizedCropRGBImageDecoder,
    )
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


class FFCVDataset(ABC):
    """
    Abstract class for FFCV datasets.
    """

    def __init__(self, dataset, write_path: str = "data/ffcv/", *args, **kwargs):
        """
        Initialize the dataset.
        """
        self.dataset = dataset
        self.write_path = write_path
        Path(self.write_path).parent.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _write(self, *args, **kwargs):
        """
        Write the dataset to disk.
        """
        pass

    @abstractmethod
    def switched_off_transform(self, *args, **kwargs):
        """
        Switch off the transform.
        """
        pass


class ImageFolderFFCV(FFCVDataset):
    """
    Helper class using FFCV for ImageFolder type datasets.
    """

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
    IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
    DEFAULT_CROP_RATIO = 224 / 256

    def __init__(
        self,
        dataset,
        write_path: str = "data/ffcv.beton",
        transform: Optional[Callable] = None,
        num_workers: int = 16,
        image_size: int = 224,
        device=default_device(),
    ):
        """
        Initialize the dataset.
        """
        super().__init__(
            dataset=dataset,
            write_path=write_path,
        )
        self.transform = transform

        with self.switched_off_transform():
            self._write(num_workers=num_workers)

        self._image_size = image_size
        self._device = device

    def _write(self, num_workers: int = 16):
        """
        Writes the dataset to the write_path.
        """
        assert ffcv, "FFCV is not installed"
        writer = DatasetWriter(
            self.write_path,
            {
                "image": RGBImageField(),
                "label": IntField(),
            },
            num_workers=num_workers,
        )

        LOGGER.info(f"Writing dataset to {self.write_path}")
        writer.from_indexed_dataset(self.dataset)

    @contextmanager
    def switched_off_transform(self):
        """
        Context manager to switch off the transforms in original dataset,
        and then switch it back on after the context is exited.
        """
        _old_transform = None
        _old_target_transform = None

        try:
            # Switch off the transforms
            if hasattr(self.dataset, "transform"):
                _old_transform = self.dataset.transform
                self.dataset.transform = None
                LOGGER.info("Transforms switched off for original dataset.")

            if hasattr(self.dataset, "target_transform"):
                _old_transform = self.dataset.target_transform
                self.dataset.transform = None
                LOGGER.info("Target Transforms switched off for original dataset.")

            # Give back control
            yield

        finally:
            # restore the transforms
            if _old_transform is not None and hasattr(self.dataset, "transform"):
                self.dataset.transform = _old_transform
                LOGGER.info("Transforms switched on for original dataset.")

            if _old_target_transform is not None and hasattr(
                self.dataset, "target_transform"
            ):
                self.dataset.target_transform = _old_target_transform
                LOGGER.info("Target Transforms switched on for original dataset.")

    def get_loader(
        self,
        num_workers: int = 16,
        batch_size: int = 32,
        distributed: int = 0,
        in_memory: int = 0,
        validation: bool = False,
    ):
        """
        Initialize the data loader.

        :param num_workers: Number of workers to use for the data loader.
        :param batch_size: Batch size for the data loader.
        :param distributed: 1/0 Whether to use distributed data loading.
        :param in_memory: 1/0 Does the dataset fit in memory.
        """

        _init_msg = "Initializing data loader."
        if validation:
            _init_msg = "Initializing validation data loader."
        LOGGER.info(_init_msg)

        resolution = (self._image_size, self._image_size)
        decoder = (
            CenterCropRGBImageDecoder(
                resolution,
                ratio=self.DEFAULT_CROP_RATIO,
            )
            if validation
            else RandomResizedCropRGBImageDecoder(resolution)
        )
        image_pipeline: List[Operation] = [
            decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(torch.device(self._device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(self.IMAGENET_MEAN, self.IMAGENET_STD, np.float32),
        ]

        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(self._device), non_blocking=True),
        ]

        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        drop_last = True

        if validation:
            # remove randomness validation step
            image_pipeline.pop(1)
            order = OrderOption.SEQUENTIAL
            drop_last = False

        loader = Loader(
            fname=self.write_path,
            batch_size=batch_size,
            num_workers=num_workers,
            order=order,
            os_cache=in_memory,
            drop_last=drop_last,
            pipelines={"image": image_pipeline, "label": label_pipeline},
            distributed=distributed,
        )

        return loader
