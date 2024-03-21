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
from typing import Optional

from transformers import PreTrainedModel
from transformers.file_utils import CONFIG_NAME

from sparseml.transformers.compression.compressors import ModelCompressor
from sparseml.transformers.compression.config import CompressionConfig
from sparseml.transformers.utils.helpers import SPARSITY_CONFIG_NAME


class SparsePreTrainedModel(PreTrainedModel):
    def save_pretrained(
        self,
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
        :param sparsity_config: optional sparsity config to compress model with, if no
        config is provided it will be inferred from the model
        :param save_compresed: whether or not to compress the model on disk
        :param kwargs: additional kwargs to pass on to PreTrainedModel.save_pretrained
        """
        if sparsity_config is not None:
            # if a sparsity config is provided, always save compressed
            sparsity_config.fill_config_details(self)
            save_compressed = True
        elif save_compressed:
            # try to infer a sparsity config from the model if none is provided
            sparsity_config = CompressionConfig.infer_config_from_model(
                self, compress=save_compressed
            )

        if sparsity_config is None:
            # model is not sparse, save as dense
            return super().save_pretrained(save_directory, **kwargs)

        # if we've gotten to this point we can run compression since we have a config
        kwargs["safe_serialization"] = True
        compressor = ModelCompressor.load_from_registry(
            sparsity_config.format, config=sparsity_config
        )

        compressed_state_dict = compressor.compress(self.state_dict())
        kwargs["state_dict"] = compressed_state_dict

        super().save_pretrained(save_directory, **kwargs)
        sparsity_config_data = sparsity_config.dict()
        config_file_path = os.path.join(save_directory, CONFIG_NAME)

        # add the sparsity config to the model's config file
        with open(config_file_path, "r") as config_file:
            config_data = json.load(config_file)
        config_data[SPARSITY_CONFIG_NAME] = sparsity_config_data
        with open(config_file_path, "w") as config_file:
            json.dump(config_data, config_file, indent=4, sort_keys=True)

    def save_compressed(
        self,
        save_directory: str,
        sparsity_config: Optional[CompressionConfig] = None,
        **kwargs,
    ):
        """
        Alias for SparseAutoModelForCausalLM.save_pretrained() that always saves in a
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
