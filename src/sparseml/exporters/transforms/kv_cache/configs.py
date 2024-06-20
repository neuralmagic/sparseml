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

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseModel, ConfigDict, Field

from sparseml.exporters.transforms import OnnxTransform
from sparseml.exporters.transforms.kv_cache.transforms_codegen import (
    AdditionalTransformsCodeGen,
)
from sparseml.exporters.transforms.kv_cache.transforms_llama import (
    AdditionalTransformsLLAMA,
)
from sparseml.exporters.transforms.kv_cache.transforms_mpt import (
    AdditionalTransformsMPT,
)
from sparseml.exporters.transforms.kv_cache.transforms_opt import (
    AdditionalTransformsOPT,
)


_LOGGER = logging.getLogger(__name__)

__all__ = ["get_kv_cache_config", "KeyValueCacheConfig"]


class KeyValueCacheConfig(BaseModel):
    model_name: str = Field(
        description="The name of the model type. This should correspond to "
        "the `model_type` field in the transformer's `config.json` file."
    )
    additional_transforms: Union[
        List[Type[OnnxTransform]], Type[OnnxTransform], None
    ] = Field(
        None,
        description="A transform class (or list thereof) to use for additional "
        "transforms to the model required for finalizing the kv cache injection.",
    )
    key_num_attention_heads: str = Field(
        description="The key to use to get the number of attention heads from the "
        "transformer's `config.json` file."
    )
    key_num_embedding_hidden_size: str = Field(
        description="The key to use to get the hidden size "
        "from the transformer's `config.json` file."
    )
    num_attention_heads: Optional[int] = Field(
        None, description="The number of attention heads."
    )
    hidden_size_kv_cache: Optional[int] = Field(
        None, description="The hidden size of the key/value cache. "
    )
    multiply_batch_by_num_att_heads: bool = Field(
        default=False,
        description="Whether or not to internally multiply "
        "the batch size by the number of attention heads. "
        "This is used to reduce the number of dimensions in "
        "the key/value cache.",
    )
    transpose_value_input: Optional[Tuple[int, int, int, int]] = Field(
        default=None,
        description="The transpose indices to apply to the value of "
        "the kv cache. If this is not provided, no transpose will "
        "be applied.",
    )
    transpose_key_input: Optional[Tuple[int, int, int, int]] = Field(
        default=None,
        description="The transpose indices to apply to the key of "
        "the kv cache. If this is not provided, no transpose will "
        "be applied.",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())


OPT_CONFIG = KeyValueCacheConfig(
    model_name="opt",
    additional_transforms=AdditionalTransformsOPT,
    key_num_attention_heads="num_attention_heads",
    key_num_embedding_hidden_size="hidden_size",
    transpose_value_input=None,
    transpose_key_input=None,
    multiply_batch_by_num_att_heads=True,
)

CODEGEN_CONFIG = KeyValueCacheConfig(
    model_name="codegen",
    additional_transforms=AdditionalTransformsCodeGen,
    key_num_attention_heads="n_head",
    key_num_embedding_hidden_size="n_embd",
    transpose_value_input=(0, 2, 1, 3),
    transpose_key_input=None,
    multiply_batch_by_num_att_heads=False,
)

# the injection config for MPT config is compatible
# with the MPT model in HF Space 'mosaicml/mpt-7b'
# at the state corresponding to the commit
# `68e1a8e0ebb9b30f3c45c1ef6195980f29063ae2`
MPT_CONFIG = KeyValueCacheConfig(
    model_name="mpt",
    additional_transforms=AdditionalTransformsMPT,
    key_num_attention_heads="n_heads",
    key_num_embedding_hidden_size="d_model",
    transpose_value_input=(0, 2, 1, 3),
    transpose_key_input=(0, 2, 1, 3),
    multiply_batch_by_num_att_heads=False,
)

BLOOM_CONFIG = KeyValueCacheConfig(
    model_name="bloom",
    additional_transforms=None,
    key_num_attention_heads="num_attention_heads",
    key_num_embedding_hidden_size="n_embed",
    transpose_value_input=None,
    transpose_key_input=(0, 1, 3, 2),
    multiply_batch_by_num_att_heads=True,
)

LLAMA_CONFIG = KeyValueCacheConfig(
    model_name="llama",
    additional_transforms=AdditionalTransformsLLAMA,
    key_num_attention_heads="num_attention_heads",
    key_num_embedding_hidden_size="hidden_size",
    transpose_value_input=(0, 2, 1, 3),
    transpose_key_input=None,
    multiply_batch_by_num_att_heads=False,
)

# Mistral has a config/model definition "MistralForCausalLM" but is based off Llama2.
# It contains these additions to Llama2-7b:
# * Sliding Window Attention
# * GQA (Grouped Query Attention)
# * Byte-fallback BPE tokenizer
MISTRAL_CONFIG = KeyValueCacheConfig(
    model_name="mistral",
    additional_transforms=AdditionalTransformsLLAMA,
    key_num_attention_heads="num_attention_heads",
    key_num_embedding_hidden_size="hidden_size",
    transpose_value_input=None,
    transpose_key_input=None,
    multiply_batch_by_num_att_heads=False,
)

# Reusing the CodeGen transforms because it happens to match what we need for GPTNeo
additional_transforms_gpt_neo = AdditionalTransformsCodeGen

GPT_NEO_CONFIG = KeyValueCacheConfig(
    model_name="gpt_neo",
    additional_transforms=additional_transforms_gpt_neo,
    key_num_attention_heads="num_heads",
    key_num_embedding_hidden_size="hidden_size",
    transpose_value_input=(0, 2, 1, 3),
    transpose_key_input=None,
    multiply_batch_by_num_att_heads=False,
)


def get_kv_cache_config(
    model_path: str,
    supported_configs: List[BaseModel] = [
        OPT_CONFIG,
        CODEGEN_CONFIG,
        BLOOM_CONFIG,
        MPT_CONFIG,
        LLAMA_CONFIG,
        MISTRAL_CONFIG,
        GPT_NEO_CONFIG,
    ],
) -> KeyValueCacheConfig:
    """
    Get the kv cache config for the model at the given path.

    :param model_path: The path to the directory containing
        the transformers model. It is assumed that
        the `config.json` file (as supplied by the
        transformers models) is in this directory.
    :param supported_configs: The list of supported configs.
        If the model type is not in this list,
        a warning will be logged and the first
        config in the list will be returned.
    :return: The kv cache config for the model.
    """
    transformers_config = _get_transformers_config(model_path)
    model_name = transformers_config["model_type"]

    kv_cache_config = [
        kv_cache_config
        for kv_cache_config in supported_configs
        if kv_cache_config.model_name == model_name
    ]
    if len(kv_cache_config) == 0:
        _LOGGER.warning(
            f"Could not find a kv cache config for model type: {model_name}."
        )
        return None

    kv_cache_config = kv_cache_config[0]

    # set the number of attention heads and the hidden size of the kv cache
    num_attention_heads = transformers_config.get(
        kv_cache_config.key_num_attention_heads
    )
    hidden_size_kv_cache = (
        transformers_config.get(kv_cache_config.key_num_embedding_hidden_size)
        // num_attention_heads
    )
    kv_cache_config.num_attention_heads = num_attention_heads
    kv_cache_config.hidden_size_kv_cache = hidden_size_kv_cache

    kv_cache_config = adapt_cache_structure_for_gqa(
        kv_cache_config, transformers_config
    )

    _LOGGER.info("Properly configured arguments for KV Cache Transformation")
    return kv_cache_config


def adapt_cache_structure_for_gqa(
    kv_cache_config: KeyValueCacheConfig,
    transformers_config: Dict[str, Any],
    model_names: List[str] = ["llama"],
) -> KeyValueCacheConfig:
    """
    Potentially adapts the kv_cache_config, so that it
    properly works with Grouped Query Attention (GQA).

    For now, this function only supports the llama model.
    Llama uses:
    Multi Head Attention (MHA) if `num_key_value_heads==num_attention_heads` (default),
    Grouped Query Attention (GQA) if `num_key_value_heads<num_attention_heads`,
    Multi Query Attention (MQA) if `num_key_value_heads==1`,

    :param kv_cache_config: The kv cache config for the model.
    :param transformers_config: The transformers config for
        the model. If contains the key:`num_key_value_heads`,
        the model may be potentially using GQA instead of
        MHA and thus the kv_cache_config needs to be adapted.
    :param model_names: The list of model names that may use
        GQA instead of MQA.
    :return: Potentially adapted kv cache config for the model.
        If the model does not use GQA, the kv_cache_config is
        returned unchanged.
    """
    # For now, we only support GQA for LLAMA.
    model_name = kv_cache_config.model_name
    num_attention_heads = kv_cache_config.num_attention_heads
    num_key_value_heads = transformers_config.get("num_key_value_heads")

    if num_key_value_heads is not None and model_name in model_names:
        if num_key_value_heads > 1 and num_key_value_heads != num_attention_heads:
            # introduce the modification the config to support GQA for LLAMA.
            kv_cache_config.transpose_value_input = None

            _LOGGER.info(
                f"Adapted the model: {transformers_config['model_type']} "
                f"to work with GQA."
            )
    return kv_cache_config


def _get_transformers_config(model_path: Union[str, Path]) -> Dict[str, Any]:
    # from the model path, get the config.json file and return it as a dict.
    model_path = Path(model_path) if isinstance(model_path, str) else model_path

    if not model_path.is_dir():
        raise ValueError(
            f"`model_path` is expected to be a directory, found {model_path}"
        )
    config_file = [file for file in model_path.iterdir() if file.name == "config.json"]
    if len(config_file) == 0:
        raise ValueError(f"Unable to find config.json in model_path: {model_path}")
    config_file = config_file[0]

    with open(config_file) as f:
        config = json.load(f)
    _LOGGER.info(f"Loaded config file {config_file} for model: {config['model_type']}")
    return config
