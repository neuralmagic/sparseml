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
import os
import weakref
from functools import wraps
from typing import Optional

from transformers import PreTrainedModel
from transformers.file_utils import CONFIG_NAME

from compressed_tensors import (
    QUANTIZATION_CONFIG_NAME,
    SPARSITY_CONFIG_NAME,
    CompressionConfig,
    ModelCompressor,
    QuantizationConfig,
)
from compressed_tensors.quantization.utils import is_model_quantized
from sparseml.transformers.compression.sparsity_config import SparsityConfigMetadata
from sparseml.utils.pytorch import qat_active


_LOGGER = logging.getLogger(__name__)

__all__ = ["modify_save_pretrained"]


def modify_save_pretrained(model: PreTrainedModel):
    """
    Overrides a PreTrainedModel's save_pretrained() method with a wrapped version that
    supports compression
    """

    def save_pretrained_compressed(save_pretrained_method):
        if getattr(save_pretrained_method, "_overridden", False):
            # `model.save_pretrained` has already been replaced, return.
            return save_pretrained_method

        # Keep a weak reference to the model class and unbound save_pretrained
        # method so we can call the original
        model_ref = weakref.ref(save_pretrained_method.__self__)
        original_save_pretrained = save_pretrained_method.__func__
        model_class = model_ref().__class__
        del save_pretrained_method

        @wraps(original_save_pretrained)
        def save_pretrained_wrapper(
            save_directory: str,
            sparsity_config: Optional[CompressionConfig] = None,
            save_compressed: bool = False,
            skip_compression_stats: bool = False,
            **kwargs,
        ):
            """
            Wrapper around PreTrainedModel.save_pretrained(), adds functionality for
            saving models in a compressed format on disk. The compression format is
            saved to the model's config file

            :param save_directory: output directory to save model to
            :param sparsity_config: optional sparsity config to compress model with,
            if no config is provided it will be inferred from the model
            :param save_compresed: whether or not to compress the model on disk
            :param skip_compression_stats: whether to skip the calculation of
            compression statistics (such as global sparsity and sparsity structure) when
            saving a model in dense format
            :param kwargs: additional kwargs to pass on to model.save_pretrained
            """
            model = model_ref()
            # state_dict gets passed in as a kwarg for FSDP models
            state_dict = kwargs.get("state_dict", None)

            # check if we are in the old quantization framework
            if qat_active(model) and not is_model_quantized(model):
                _LOGGER.info(
                    "Compression for models quantized with QuantizationModifer is not "
                    "supported. Save will be run without compression and no sparsity "
                    "statistics will be calculated. To save a quantized model in a "
                    "compressed state please use vLLMQuantizationModifier instead."
                )

                original_save_pretrained.__get__(model, model_class)(
                    save_directory, **kwargs
                )

                return

            elif qat_active(model):  # quantized in new framework
                _LOGGER.info(
                    "Sparsity compression for quantized models is not yet supported. "
                    "No sparsity statistics will be calculated and no sparsity config "
                    "will be saved."
                )

                original_save_pretrained.__get__(model, model_class)(
                    save_directory, **kwargs
                )

                quant_config = QuantizationConfig.from_pretrained(model)
                quant_config_data = quant_config.model_dump(exclude_unset=True)
                config_file_path = os.path.join(save_directory, CONFIG_NAME)

                # add the sparsity config to the model's config file
                with open(config_file_path, "r") as config_file:
                    config_data = json.load(config_file)
                config_data[QUANTIZATION_CONFIG_NAME] = quant_config_data
                with open(config_file_path, "w") as config_file:
                    json.dump(config_data, config_file, indent=2, sort_keys=True)

                return

            if sparsity_config is not None:
                sparsity_config.global_sparsity = (
                    SparsityConfigMetadata.infer_global_sparsity(
                        model, state_dict=state_dict
                    )
                )
                sparsity_config.sparsity_structure = (
                    SparsityConfigMetadata.infer_sparsity_structure()
                )

            elif not skip_compression_stats:
                # try to infer a sparsity config from the model if none is provided
                _LOGGER.info(
                    "Inferring a sparsity configuration requires a global sparsity "
                    "calculation. This can be costly for large models. To skip the "
                    "calculation of compression statistics set "
                    "skip_compression_stats=True"
                )
                sparsity_config = SparsityConfigMetadata.from_pretrained(
                    model, state_dict=state_dict, compress=save_compressed
                )

            if sparsity_config is None:
                # model is not sparse, save as dense
                return original_save_pretrained.__get__(model, model_class)(
                    save_directory, **kwargs
                )

            # if we've gotten to this point we have a config so we can run compression
            kwargs["safe_serialization"] = True
            compressor = ModelCompressor.load_from_registry(
                sparsity_config.format, config=sparsity_config
            )

            if state_dict is None:
                state_dict = model.state_dict()

            # make sure we're on the main process when saving
            if state_dict is not None and len(state_dict) > 0:
                compressed_state_dict = compressor.compress(state_dict)
                kwargs["state_dict"] = compressed_state_dict

                original_save_pretrained.__get__(model, model_class)(
                    save_directory, **kwargs
                )
                sparsity_config_data = sparsity_config.dict()
                config_file_path = os.path.join(save_directory, CONFIG_NAME)

                # add the sparsity config to the model's config file
                with open(config_file_path, "r") as config_file:
                    config_data = json.load(config_file)
                config_data[SPARSITY_CONFIG_NAME] = sparsity_config_data
                with open(config_file_path, "w") as config_file:
                    json.dump(config_data, config_file, indent=2, sort_keys=True)

        save_pretrained_wrapper._overriden = True
        return save_pretrained_wrapper

    # wrap save_pretrained
    model.save_pretrained = save_pretrained_compressed(model.save_pretrained)
