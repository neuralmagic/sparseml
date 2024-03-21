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
import os
import types
import weakref
from functools import wraps
from typing import Optional

from transformers import AutoConfig
from transformers.file_utils import CONFIG_NAME

from sparseml.transformers.compression.compressors import ModelCompressor
from sparseml.transformers.compression.config import CompressionConfig
from sparseml.transformers.utils.helpers import SPARSITY_CONFIG_NAME


__all__ = [
    "infer_compressor_from_model_config",
    "modify_save_pretrained",
    "add_save_compressed_method",
]


def infer_compressor_from_model_config(
    pretrained_model_name_or_path: str,
) -> Optional[ModelCompressor]:
    """
    Given a path to a model config, extract a sparsity config if it exists and return
    the associated ModelCompressor

    :param pretrained_model_name_or_path: path to model config on disk or HF hub
    :return: matching compressor if config contains a sparsity config
    """
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    sparsity_config = getattr(config, SPARSITY_CONFIG_NAME, None)
    if sparsity_config is None:
        return None

    format = sparsity_config.get("format")
    sparsity_config = CompressionConfig.load_from_registry(format, **sparsity_config)
    compressor = ModelCompressor.load_from_registry(format, config=sparsity_config)
    return compressor


def modify_save_pretrained(model):
    def save_pretrained_compressed(save_pretrained_method):
        if getattr(save_pretrained_method, "_overridden", False):
            # `model.save_pretrained` has already been replaced, return.
            return save_pretrained_method

        # Prevent cyclic references by keeping a weak reference
        # to optimizer class and original unbound step method
        model_ref = weakref.ref(save_pretrained_method.__self__)
        original_save_pretrained = save_pretrained_method.__func__
        model_class = model_ref().__class__
        del save_pretrained_method

        @wraps(original_save_pretrained)
        def save_pretrained_wrapper(
            save_directory: str,
            sparsity_config: Optional[CompressionConfig] = None,
            save_compressed: bool = False,
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
            :param kwargs: additional kwargs to pass on to model.save_pretrained
            """
            model = model_ref()

            if sparsity_config is not None:
                # if a sparsity config is provided, always save compressed
                sparsity_config.fill_config_details(model)
                save_compressed = True
            elif save_compressed:
                # try to infer a sparsity config from the model if none is provided
                sparsity_config = CompressionConfig.infer_config_from_model(
                    model, compress=save_compressed
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

            compressed_state_dict = compressor.compress(model.state_dict())
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
                json.dump(config_data, config_file, indent=4, sort_keys=True)

        # Note that the returned function here is no longer a bound method,
        # so attributes like `__func__` and `__self__` no longer exist.
        save_pretrained_wrapper._overriden = True
        return save_pretrained_wrapper

    # wrap save_pretrained
    model.save_pretrained = save_pretrained_compressed(model.save_pretrained)


def add_save_compressed_method(model):
    def save_compressed(
        self,
        save_directory: str,
        sparsity_config: Optional[CompressionConfig] = None,
        **kwargs,
    ):
        """
        Alias for PreTrainedModel.save_pretrained() that always saves in a
        compressed format

        :param save_directory: output directory to save model to
        :param sparsity_config: optional sparsity config to compress model with, if no
        config is provided it will be inferred from the model
        :param kwargs: additional kwargs to pass on to model.save_pretrained
        """
        return self.save_pretrained(
            save_directory=save_directory,
            sparsity_config=sparsity_config,
            save_compressed=True,
            **kwargs,
        )

    model.save_compressed = types.MethodType(save_compressed, model)
