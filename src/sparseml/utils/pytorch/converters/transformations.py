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
# flake8: noqa: F821

import functools
import logging
from typing import Dict

import numpy
import numpy as np
import torch
from torch import Tensor


_LOGGER = logging.getLogger(__name__)


def _log_transformation(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _LOGGER.info("Applying transformation: %s", func.__name__.upper())
        return_value = func(*args, **kwargs)
        _LOGGER.info("Transformation: %s complete", func.__name__.upper())
        return return_value

    return wrapper


def is_gptq_quantization_target(key: str) -> bool:
    """
    Assumes self_attn and mlp are the only quantization targets
    in model layers of the state_dict.
    :param key: The key of the state_dict
    :return: True if the key is a quantization target, False otherwise
    """
    return "model.layers" in key and ("self_attn" in key or "mlp" in key)


@_log_transformation
def transform_exllama_names(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """
    Transforms the exallama state_dict keys to be compatible with
    SparseAutoModel classes.

    The renames include:
        - scales -> weight_fake_quant.scale
        - qzeros -> weight_fake_quant.zero_point
        - qweight -> weight

    Note: does not transforms the actual tensor values

    :pre-condition: The state_dict should be for a quantized model
    :pre-condition: Targets only the weights of the self_attn and mlp nodes
    :param state_dict: The quantized state_dict to be transformed
    :return: The transformed state_dict
    """

    name_map: Dict[str, str] = {
        ".scales": ".weight_fake_quant.scale",
        ".qzeros": ".weight_fake_quant.zero_point",
        ".qweight": ".weight",
    }

    updated_state_dict = {}
    for key, tensor in state_dict.items():
        if any(key.endswith(target_suffix := suffix) for suffix in name_map):
            updated_key = key.replace(target_suffix, name_map[target_suffix])
            updated_state_dict[updated_key] = tensor
        else:
            updated_state_dict[key] = tensor
    return updated_state_dict


@_log_transformation
def transform_autogptq_weights_and_reshape_tensors(
    state_dict: Dict[str, Tensor]
) -> Dict[str, Tensor]:
    """
    Tranforms weights into their required shapes and types for Exllama
    to CompressedTensors conversion

    The transformations include:
        - Unpack ad dequantize the weight tensor using the scales, zeros, and g_idx tensors
        - Squeeze the scales tensor to [x] from [1, x]

    :pre-condition: The state_dict should be for a quantized model
    :pre-condition: The state_dict should have the bias and g_idx tensors added

    :param state_dict: The state_dict to be transformed
    :return: The transformed state_dict, with repacked and reshaped tensors
    """

    transformed_state_dict: Dict[str, Tensor] = {}

    # auxillary dict to store transformed weights
    transformed_weights_dict: Dict[str, Tensor] = {}

    # quantize qweights before scales, and qzeros
    # because the ordering in which tensors are fetched
    # is not guaranteed by our implementation
    for key, tensor in state_dict.items():
        if is_gptq_quantization_target(key) and key.endswith(".qweight"):
            # quantize the weight tensor
            scales = state_dict[key.replace("qweight", "scales")]
            qzeros = state_dict[key.replace("qweight", "qzeros")]
            g_idx = state_dict[key.replace("qweight", "g_idx")]

            zeros = unpack_zeros(qzeros)
            qweight = unpack_int32_into_fp32(
                qweight=tensor,
                scales=scales,
                zeros=zeros,
                g_idx=g_idx,
            )
            transformed_weights_dict[key] = qweight

    # transform scales
    for key, tensor in state_dict.items():
        if is_gptq_quantization_target(key) and key.endswith(".scales"):
            # scales [1, x] should be reshaped to [x]
            scales = tensor.squeeze(0)
            transformed_state_dict[key] = scales
        else:
            transformed_state_dict[key] = tensor

    # overwrite old weights with the new quantized weights
    transformed_state_dict.update(transformed_weights_dict)

    # auxillary weights_dict not needed anymore
    del transformed_weights_dict

    return transformed_state_dict


def unpack_zeros(qzeros):
    """
    Unpack the quantized zero points tensor from 32 bit integers into 4 bit integers.

    :param qzeros: The quantized zero points tensor of int32 dtype and shape [1, 8x]
    """
    bits = 4
    qzeros = qzeros.numpy().astype(np.uint32)
    intzeros = np.zeros(
        (qzeros.shape[0], qzeros.shape[1] * 32 // bits), dtype=np.uint32
    )

    i = 0
    col = 0
    while col < intzeros.shape[1]:
        if bits in [4]:
            for j in range(i, min(i + (32 // bits), intzeros.shape[1])):
                intzeros[:, j] = (qzeros[:, col] >> (bits * (j - i))) & 0xF
            i += 32 // bits
            col += 1
        else:
            raise NotImplementedError("Only 4 bits are supported.")

    intzeros = intzeros.astype(np.int32)
    intzeros = torch.from_numpy(intzeros)

    return intzeros


def unpack_int32_into_fp32(
    qweight: Tensor, scales: Tensor, zeros: Tensor, g_idx: Tensor
) -> Tensor:
    """
    Unpack the quantized weight tensor from 32 bit integers into 4 bit integers,
    and then dequantize them using the scales, zeros, and g_idx tensors.

    :param qweight: The quantized weight tensor of int32 dtype and shape [x, y]
    :param scales: The scales tensor
    :param zeros: The zero points tensor
    :param g_idx: The group index tensor
    :return: The dequantized weight tensor of shape [x, 8y]
    """
    bits = 4
    qweight = qweight.numpy().astype(numpy.uint32)
    intweight = numpy.zeros(
        (qweight.shape[0] * 32 // bits, qweight.shape[1]), dtype=numpy.uint32
    )

    i = 0
    row = 0
    while row < intweight.shape[0]:
        if bits in [4]:
            for j in range(i, min(i + (32 // bits), intweight.shape[0])):
                intweight[j] = (qweight[row] >> (bits * (j - i))) & 0xF
            i += 32 // bits
            row += 1
        else:
            raise NotImplementedError("Only 4 bits are supported.")

    intweight = torch.from_numpy(intweight.astype(numpy.int32))
    intweight = intweight.t().contiguous()

    scales = scales.t().contiguous()
    zeros = zeros.t().contiguous()
    scale_zeros = zeros * scales
    scales = scales.clone().half()

    weight = []
    infeatures = intweight.shape[1]
    for idx in range(infeatures):
        weight.append(
            (
                intweight[:, idx].float() * scales[:, g_idx[idx]]
                - scale_zeros[:, g_idx[idx]]
            )[:, None]
        )
    weight = torch.cat(weight, dim=1)

    return weight
