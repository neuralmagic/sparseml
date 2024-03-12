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
from typing import Dict

import numpy
import torch
from torch.nn import Module
from tqdm import tqdm

from sparseml.transformers.compression.compressors import ModelCompressor


__all__ = [
    "BitmaskCompressor",
    "NumpyBitmaskTensor",
    "pack_bitmasks",
    "unpack_bitmasks",
    "bitmask_compress",
    "bitmask_decompress",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)


@ModelCompressor.register(name="sparse_bitmask")
class BitmaskCompressor(ModelCompressor):
    def compress(self, model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        compressed_dict = {}
        _LOGGER.info(f"Compressing model with {len(model_state)} weights...")
        for name, value in tqdm(model_state.items()):
            bitmask_tensor = NumpyBitmaskTensor(value)
            compressed_dict |= bitmask_tensor.dict(name_prefix=name)

        return compressed_dict

    def uncompress(self, model: Module, safetensors_path: str) -> Dict:
        raise NotImplementedError()


class NumpyBitmaskTensor:
    def __init__(self, tensor: torch.Tensor):
        self.shape = tensor.shape
        self.values, self.bitmasks, self.row_offsets = bitmask_compress(tensor.cpu())

    def decompress(self) -> torch.Tensor:
        return bitmask_decompress(self.values, self.bitmasks, self.shape)

    def to_dense(self) -> torch.Tensor:
        return self.decompress()

    @staticmethod
    def from_dense(tensor: torch.Tensor) -> "NumpyBitmaskTensor":
        return NumpyBitmaskTensor(tensor)
    
    def curr_memory_size_bytes(self):
        def sizeof_tensor(a):
            return a.element_size() * a.nelement()

        return (
            sizeof_tensor(self.values)
            + sizeof_tensor(self.bitmasks)
            + sizeof_tensor(self.row_offsets)
        )

    def __repr__(self):
        return f"NumpyBitmaskTensor(shape={self.shape}, compressed=True)"

    def dict(self, name_prefix: str) -> Dict[str, torch.Tensor]:
        return {
            name_prefix + ".compressed": self.values,
            name_prefix + ".bitmask": self.bitmasks,
            name_prefix + ".shape": torch.tensor(self.shape, device="cpu"),
            name_prefix + ".row_offsets": self.row_offsets,
        }


def pack_bitmasks(bitmasks: torch.Tensor) -> torch.Tensor:
    packed_bits_numpy = numpy.packbits(bitmasks.numpy(), axis=-1, bitorder="little")
    packed_bits_torch = torch.from_numpy(packed_bits_numpy)

    return packed_bits_torch


def unpack_bitmasks(
    packed_bitmasks: torch.Tensor, original_shape: torch.Size
) -> torch.Tensor:
    # Unpack the bits
    unpacked_bits = numpy.unpackbits(
        packed_bitmasks.numpy(), axis=-1, count=original_shape[-1], bitorder="little"
    )

    # Reshape to match the original shape
    unpacked_bitmasks_torch = torch.from_numpy(
        unpacked_bits.reshape(original_shape).astype(bool)
    )

    return unpacked_bitmasks_torch


def bitmask_compress(tensor: torch.Tensor):
    bytemasks = tensor != 0
    row_counts = bytemasks.sum(dim=-1)
    row_offsets = torch.cumsum(row_counts, 0) - row_counts
    values = tensor[bytemasks]
    bitmasks_packed = pack_bitmasks(bytemasks)

    return values, bitmasks_packed, row_offsets


def bitmask_decompress(
    values: torch.Tensor, bitmasks: torch.Tensor, original_shape: torch.Size
) -> torch.Tensor:
    bytemasks_unpacked = unpack_bitmasks(bitmasks, original_shape)

    decompressed_tensor = torch.zeros(original_shape, dtype=values.dtype)
    decompressed_tensor[bytemasks_unpacked] = values

    return decompressed_tensor
