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
from typing import Dict, Generator, List, Tuple, Union

import numpy
import torch
from torch import Tensor
from tqdm import tqdm

from safetensors import safe_open
from sparseml.transformers.compression.compressors import ModelCompressor
from sparseml.transformers.compression.utils import (
    get_nested_weight_mappings,
    merge_names,
)


__all__ = [
    "BitmaskCompressor",
    "BitmaskTensor",
    "bitmask_compress",
    "bitmask_decompress",
    "pack_bitmasks",
    "unpack_bitmasks",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)


@ModelCompressor.register(name="sparse_bitmask")
class BitmaskCompressor(ModelCompressor):
    """
    Compression for sparse models using bitmasks. Non-zero weights are stored in a 1d
    values tensor, with their locations stored in a 2d bitmask
    """

    COMPRESSION_PARAM_NAMES = ["shape", "compressed", "bitmask", "row_offsets"]

    def compress(self, model_state: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Compresses a dense state dict using bitmask compression

        :param model_state: state dict of uncompressed model
        :return: compressed state dict
        """
        compressed_dict = {}
        _LOGGER.debug(
            f"Compressing model with {len(model_state)} parameterized layers..."
        )
        for name, value in tqdm(model_state.items(), desc="Compressing model"):
            bitmask_tensor = BitmaskTensor.from_dense(value)
            bitmask_dict = bitmask_tensor.dict(name_prefix=name, device="cpu")
            for key in bitmask_dict.keys():
                if key in compressed_dict:
                    _LOGGER.warn(
                        f"Expected all compressed state_dict keys to be unique, but "
                        f"found an existing entry for {key}. The existing entry will "
                        "be replaced."
                    )
            compressed_dict |= bitmask_dict

        return compressed_dict

    def decompress(self, model_path: str) -> Generator[Tuple[str, Tensor], None, None]:
        """
        Reads a bitmask compressed state dict located at model_path and returns a
        generator for sequentially decompressing back to a dense state dict

        :param model_path: path to compressed safetensors model
        :return: iterator for generating decompressed weights
        """
        weight_mappings = get_nested_weight_mappings(
            model_path, self.COMPRESSION_PARAM_NAMES
        )
        for weight_name in weight_mappings.keys():
            weight_data = {}
            for param_name, safe_path in weight_mappings[weight_name].items():
                full_name = merge_names(weight_name, param_name)
                with safe_open(safe_path, framework="pt", device="cpu") as f:
                    weight_data[param_name] = f.get_tensor(full_name)
            data = BitmaskTensor(**weight_data)
            decompressed = data.decompress()
            yield weight_name, decompressed


class BitmaskTensor:
    """
    Owns compressions and decompression for a single bitmask compressed tensor.
    Adapted from: https://github.com/mgoin/torch_bitmask/tree/main

    :param shape: shape of dense tensor
    :compressed: flat tensor of non-zero values
    :bitmask: 2d bitmask of non-zero values
    :row_offsets: flat tensor indicating what index in values each dense row starts at
    """

    def __init__(
        self,
        shape: Union[torch.Size, List],
        compressed: Tensor,
        bitmask: Tensor,
        row_offsets: Tensor,
    ):
        self.shape = list(shape)
        self.compressed = compressed
        self.bitmask = bitmask
        self.row_offsets = row_offsets

    @staticmethod
    def from_dense(tensor: Tensor) -> "BitmaskTensor":
        """
        :param tensor: dense tensor to compress
        :return: instantiated compressed tensor
        """
        shape = tensor.shape
        compressed, bitmask, row_offsets = bitmask_compress(tensor.cpu())
        return BitmaskTensor(
            shape=shape, compressed=compressed, bitmask=bitmask, row_offsets=row_offsets
        )

    def decompress(self) -> Tensor:
        """
        :return: reconstructed dense tensor
        """
        return bitmask_decompress(self.compressed, self.bitmask, self.shape)

    def curr_memory_size_bytes(self):
        """
        :return: size in bytes required to store compressed tensor on disk
        """

        def sizeof_tensor(a):
            return a.element_size() * a.nelement()

        return (
            sizeof_tensor(self.compressed)
            + sizeof_tensor(self.bitmask)
            + sizeof_tensor(self.row_offsets)
        )

    def dict(self, name_prefix: str, device: str = "cpu") -> Dict[str, Tensor]:
        """
        :name_prefix: name of original tensor to store compressed weight as
        :return: dict of compressed data for the stored weight
        """
        return {
            merge_names(name_prefix, "shape"): torch.tensor(self.shape, device=device),
            merge_names(name_prefix, "compressed"): self.compressed.to(device),
            merge_names(name_prefix, "bitmask"): self.bitmask.to(device),
            merge_names(name_prefix, "row_offsets"): self.row_offsets.to(device),
        }

    def __repr__(self):
        return f"BitmaskTensor(shape={self.shape}, compressed=True)"


def bitmask_compress(tensor: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compresses a dense tensor using bitmask compression

    :param tensor: dense tensor to compress
    :return: tuple of compressed data representing tensor
    """
    bytemasks = tensor != 0
    row_counts = bytemasks.sum(dim=-1)
    row_offsets = torch.cumsum(row_counts, 0) - row_counts
    values = tensor[bytemasks]
    bitmasks_packed = pack_bitmasks(bytemasks)

    return values, bitmasks_packed, row_offsets


def bitmask_decompress(
    values: Tensor, bitmasks: Tensor, original_shape: torch.Size
) -> Tensor:
    """
    Reconstructs a dense tensor from a compressed one

    :param values: 1d tensor of non-zero values
    :param bitmasks: 2d int8 tensor flagging locations of non-zero values in the
    tensors original shape
    :param original_shape: shape of the dense tensor
    :return: decompressed dense tensor
    """
    bytemasks_unpacked = unpack_bitmasks(bitmasks, original_shape)

    decompressed_tensor = torch.zeros(original_shape, dtype=values.dtype)
    decompressed_tensor[bytemasks_unpacked] = values

    return decompressed_tensor


def pack_bitmasks(bytemasks: Tensor) -> Tensor:
    """
    Converts a bytemask tensor to a bitmask tensor to reduce memory. Shape RxC will be
    compressed to R x ceil(C/8)
    :param bytemasks: mask tensor where each byte corresponds to a weight
    :return: mask tensor where each bit corresounds to a weight
    """
    packed_bits_numpy = numpy.packbits(bytemasks.numpy(), axis=-1, bitorder="little")
    packed_bits_torch = torch.from_numpy(packed_bits_numpy)

    return packed_bits_torch


def unpack_bitmasks(packed_bitmasks: Tensor, original_shape: torch.Size) -> Tensor:
    """
    Converts a bitmask tensor back to a bytemask tensor for use during decompression

    :param packed_bitmasks: mask tensor where each bit corresponds to a weight
    :param original_shape: dense shape to decompress to
    :return: boolean mask of weights in the original dense shape
    """
    # Unpack the bits
    unpacked_bits = numpy.unpackbits(
        packed_bitmasks.numpy(), axis=-1, count=original_shape[-1], bitorder="little"
    )

    # Reshape to match the original shape
    unpacked_bitmasks_torch = torch.from_numpy(
        unpacked_bits.reshape(original_shape).astype(bool)
    )

    return unpacked_bitmasks_torch
