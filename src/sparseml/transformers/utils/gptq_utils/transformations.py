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
# flake8: noqa #F821,#E501

import functools
import logging
from typing import Callable, Dict, List, Tuple

import numpy
import torch
from torch import Tensor


__all__ = [
    "transform_to_exllama_names",
    "add_exllama_tensors",
    "transform_gptq_weights_and_reshape_tensors",
    "remove_unwanted_tensors_for_exllama",
    "is_gptq_quantization_target",
    "convert_fp32_tensors_to_fp16",
    "gptq_exllama_transformations",
    "GPTQ_EXLLAMA_TRANSFORMATIONS",
]

_LOGGER = logging.getLogger(__name__)

TransformationType = Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]


def _log_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _LOGGER.debug("Applying transformation: %s", func.__name__.upper())
        return_value = func(*args, **kwargs)
        _LOGGER.debug("Transformation: %s complete", func.__name__.upper())
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


@_log_call
def transform_to_exllama_names(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """
    Transforms the state_dict keys to match with exllama format

    The renames include:
        - weight_fake_quant.scale -> scales
        - weight_fake_quant.zero_point -> qzeros
        - weight -> qweight

    Note: does not transforms the actual tensor values

    :pre-condition: The state_dict should be for a quantized model
    :pre-condition: Targets only the weights of the self_attn and mlp nodes
    :param state_dict: The quantized state_dict to be transformed
    :return: The transformed state_dict
    """
    # mapping of the old names to the new names
    name_map: Dict[str, str] = {
        ".weight_fake_quant.scale": ".scales",
        ".weight_fake_quant.zero_point": ".qzeros",
        ".weight": ".qweight",
    }

    updated_state_dict: Dict[str, Tensor] = {}
    for key, tensor in state_dict.items():
        if is_gptq_quantization_target(key) and any(
            key.endswith(target_suffix := suffix) for suffix in name_map
        ):
            updated_key = key.replace(target_suffix, name_map[target_suffix])
            updated_state_dict[updated_key] = tensor
        else:
            updated_state_dict[key] = tensor
    return updated_state_dict


@_log_call
def add_exllama_tensors(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """
    Add the bias and g_idx tensors to the state_dict, wherever
    a qweight tensor is present

    The added tensors include:
        - bias: A tensor of shape [output channels] filled with zeros
                and dtype float16
        - g_idx: A tensor of shape [num_channels] filled with zeros
                and dtype int32

    :pre-condition: The state_dict should be for a quantized model
    :pre-condition: The state_dict should have been transformed to exllama names
    :param state_dict: The state_dict to be transformed
    :return: The state_dict with the added bias and g_idx tensors
    """

    updated_state_dict: Dict[str, Tensor] = {}

    for key, tensor in state_dict.items():
        if is_gptq_quantization_target(key) and key.endswith(".qweight"):
            # add bias and g_idx tensors
            bias_key = key.replace(".qweight", ".bias")
            g_idx_key = key.replace(".qweight", ".g_idx")

            # bias tensor
            bias_tensor = torch.zeros(tensor.shape[0], dtype=torch.float16)
            updated_state_dict[bias_key] = bias_tensor

            # g_idx tensor of shape [num_channels] dtype int32 filled
            # with zeros
            g_idx_tensor = torch.zeros(tensor.shape[1], dtype=torch.int32)
            updated_state_dict[g_idx_key] = g_idx_tensor

        # copy the original tensor, (qweight is also copied in this step)
        updated_state_dict[key] = tensor
    return updated_state_dict


@_log_call
def transform_gptq_weights_and_reshape_tensors(
    state_dict: Dict[str, Tensor]
) -> Dict[str, Tensor]:
    """
    Tranforms weights into their required shapes and types for Exllama format

    The transformations include:
        - Quantize the weight tensor using the scales, zeros, and g_idx tensors
            additonally pack a group of 8 of them into a single 32 bit integer
            and rename the tensor to qweight
        - Reshape the scales tensor to [1, x] and convert to fp16
        - Reshape the zero points tensor to [1, x] of type int32 and fill with zeros
            (it is assumed that quantization was symmetric)

    :pre-condition: The state_dict should be for a quantized model
    :pre-condition: The state_dict should have been transformed to exllama names
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
            qweight = _pack_fp32_into_int32(
                weight=tensor,
                scales=state_dict[key.replace("qweight", "scales")],
                zeros=state_dict[key.replace("qweight", "qzeros")],
                g_idx=state_dict[key.replace("qweight", "g_idx")],
            )
            assert qweight.dtype == torch.int32
            transformed_weights_dict[key] = qweight

    # transform scales and zero points
    for key, tensor in state_dict.items():
        if is_gptq_quantization_target(key) and key.endswith(".scales"):
            # scales [x] should be reshaped to [1, x]
            # and converted to fp16
            scales = tensor.reshape(1, -1).half()
            transformed_state_dict[key] = scales
        elif is_gptq_quantization_target(key) and key.endswith(".qzeros"):
            # zero points [8x] should be reshaped to [1, x]
            # of type int32 and filled with zeros (symmetric quantization)
            zeros = torch.zeros(tensor.shape[0] // 8, dtype=torch.int32)
            transformed_state_dict[key] = zeros.reshape(1, -1)
        else:
            transformed_state_dict[key] = tensor

    # overwrite old weights with the new quantized weights
    transformed_state_dict.update(transformed_weights_dict)

    # auxillary weights_dict not needed anymore
    del transformed_weights_dict

    return transformed_state_dict


@_log_call
def remove_unwanted_tensors_for_exllama(
    state_dict: Dict[str, Tensor]
) -> Dict[str, Tensor]:
    """
    Remove unwanted tensors from the state_dict that are not necessary for inference.
    These tensors include:
        - eps
        - min_val
        - max_val
        - fake_quant_enabled
        - observer_enabled

    :param state_dict: The state_dict to be cleaned
    :return: The cleaned state_dict with all keys ending with the unwanted suffixes removed
    """
    suffixes_to_delete: List[str] = [
        "eps",
        "min_val",
        "max_val",
        "fake_quant_enabled",
        "observer_enabled",
    ]
    keys = list(state_dict.keys())
    for key in keys:
        if any(key.endswith(suffix) for suffix in suffixes_to_delete):
            del state_dict[key]
    return state_dict


@_log_call
def convert_fp32_tensors_to_fp16(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """
    Convert all remaining fp32 tensors to fp16 tensors in the state_dict
    This is expected by the Exllama format

    :param state_dict: The state_dict to be converted
    :return: The converted state_dict, with all fp32 tensors converted to fp16
    """
    converted_state_dict: Dict[str, Tensor] = {}
    for key, tensor in state_dict.items():
        converted_state_dict[key] = (
            tensor.half() if tensor.dtype == torch.float32 else tensor
        )
    return converted_state_dict


def gptq_exllama_transformations() -> Tuple[TransformationType, ...]:
    """
    :return: An Iterable of transformations that must be applied to
        the state_dict IN_ORDER to convert it to the Exllama format
        for GPTQ style quantization. Each transformation is a
        callable that accepts a state_dict and returns a transformed
        state_dict.
    """

    return (
        transform_to_exllama_names,
        add_exllama_tensors,
        transform_gptq_weights_and_reshape_tensors,
        remove_unwanted_tensors_for_exllama,
        convert_fp32_tensors_to_fp16,
    )


def _pack_fp32_into_int32(
    weight: Tensor, scales: Tensor, zeros: Tensor, g_idx: Tensor
) -> Tensor:
    """
    Quantize the weight tensor using the scales, zeros, and g_idx tensors
    into 4 bit integers, and packs a group of 8 of them into a single 32 bit integer.

    Adapted from:
    https://github.com/AutoGPTQ/AutoGPTQ/blob/ea4a99778f90b60c9b5177d7487af1b4ca87744f/auto_gptq/nn_modules/qlinear/qlinear_exllama.py#L118

    :param weight: The weight tensor to be quantized of shape [x, 8y]
    :param scales: The scales tensor
    :param zeros: The zero points tensor
    :param g_idx: The group index tensor
    :return: The quantized weight tensor of int32 dtype and shape [x, y]
    """
    g_idx = g_idx.clone()

    scales = scales.t().contiguous()
    zeros = zeros.t().contiguous()
    scale_zeros = zeros * scales
    scales = scales.clone().half()
    bits = 4

    intweight = []
    infeatures = weight.shape[1]
    for idx in range(infeatures):
        intweight.append(
            torch.round(
                (weight[:, idx] + scale_zeros[g_idx[idx]]) / scales[g_idx[idx]]
            ).to(torch.int)[:, None]
        )
    intweight = torch.cat(intweight, dim=1)
    intweight = intweight.t().contiguous()
    intweight = intweight.numpy().astype(numpy.uint32)

    i = 0
    row = 0
    qweight = numpy.zeros(
        (intweight.shape[0] // 32 * bits, intweight.shape[1]), dtype=numpy.uint32
    )
    while row < qweight.shape[0]:
        if bits in [4]:
            for j in range(i, i + (32 // bits)):
                qweight[row] |= intweight[j] << (bits * (j - i))
            i += 32 // bits
            row += 1
        else:
            raise NotImplementedError("Only 4 bits are supported.")

    qweight = qweight.astype(numpy.int32)
    qweight = torch.from_numpy(qweight)
    return qweight


GPTQ_EXLLAMA_TRANSFORMATIONS: Tuple[
    TransformationType, ...
] = gptq_exllama_transformations()
