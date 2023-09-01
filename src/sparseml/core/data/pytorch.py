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

import torch

from sparseml.core.data.base import ModifiableData
from torch.utils.data import DataLoader
from typing import Mapping, Sequence

__all__ = ["ModifiableDataPyTorch", "DynamicBatchSizeDataLoader"]


class DynamicBatchSizeDataLoader:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.current_batch_size = data_loader.batch_size

    def __iter__(self):
        if self.current_batch_size == self.data_loader.batch_size:
            yield from self.data_loader
        elif self.current_batch_size < self.data_loader.batch_size:
            yield from self._data_split_iter()
        else:
            yield from self._data_merge_iter()

    def set_batch_size(self, batch_size: int):
        self.current_batch_size = batch_size

    def get_batch_size(self) -> int:
        return self.current_batch_size

    def _data_split_iter(self):
        if self.current_batch_size >= self.data_loader.batch_size:
            raise ValueError(
                "Current batch size must be less than the original batch size"
            )

        for batch in self.data_loader:
            num_splits = self.data_loader.batch_size // self.current_batch_size
            for i in range(num_splits):
                start_idx = i * self.current_batch_size
                end_idx = (i + 1) * self.current_batch_size
                yield DynamicBatchSizeDataLoader.split_batch(batch, start_idx, end_idx)

    def _data_merge_iter(self):
        if self.current_batch_size <= self.data_loader.batch_size:
            raise ValueError(
                "Current batch size must be greater than the original batch size"
            )

        buffer = []
        buffer_size = 0
        for batch in self.data_loader:
            buffer.append(batch)
            buffer_size += len(batch)
            while buffer_size >= self.current_batch_size:
                merged = DynamicBatchSizeDataLoader.merge_batches(buffer)
                yield DynamicBatchSizeDataLoader.split_batch(
                    merged, 0, self.current_batch_size
                )
                buffer = [
                    DynamicBatchSizeDataLoader.split_batch(
                        merged, self.current_batch_size, buffer_size
                    )
                ]
                buffer_size -= self.current_batch_size

    @staticmethod
    def split_batch(batch, start_idx, end_idx):
        """
        Splits a batch based on its type (Tensor, Mapping, Sequence) and the provided indices.
        """
        if isinstance(batch, torch.Tensor):
            return batch[start_idx:end_idx]
        elif isinstance(batch, Mapping):
            return {
                key: DynamicBatchSizeDataLoader.split_batch(value, start_idx, end_idx)
                for key, value in batch.items()
            }
        elif isinstance(batch, Sequence):
            return [
                DynamicBatchSizeDataLoader.split_batch(item, start_idx, end_idx)
                for item in batch
            ]
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

    @staticmethod
    def merge_batches(batches):
        """
        Merges a sequence of batches into a single batch.
        """
        sample_batch = batches[0]
        if isinstance(sample_batch, torch.Tensor):
            return torch.cat(batches, dim=0)
        elif isinstance(sample_batch, Mapping):
            return {
                key: DynamicBatchSizeDataLoader.merge_batches(
                    [batch[key] for batch in batches]
                )
                for key in sample_batch.keys()
            }
        elif isinstance(sample_batch, Sequence):
            return [
                DynamicBatchSizeDataLoader.merge_batches(
                    [batch[i] for batch in batches]
                )
                for i in range(len(sample_batch))
            ]
        else:
            raise TypeError(f"Unsupported batch type: {type(sample_batch)}")


class ModifiableDataPyTorch(ModifiableData[DynamicBatchSizeDataLoader]):
    def __init__(self, data_loader: DataLoader):
        super().__init__()
        self.data = DynamicBatchSizeDataLoader(data_loader)

    def get_num_batches(self) -> int:
        return self.num_samples // self.data.get_batch_size()

    def set_batch_size(self, batch_size: int):
        self.data.set_batch_size(batch_size)

    def get_batch_size(self) -> int:
        return self.data.get_batch_size()
