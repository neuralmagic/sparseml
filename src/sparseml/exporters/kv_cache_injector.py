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
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Union

import onnx

from sparseml.exporters.base_exporter import BaseExporter
from sparseml.exporters.transforms.kv_cache import (
    CacheKeysAndValues,
    PositionEmbeddingsAdjustment,
)
from sparsezoo.utils import save_onnx


_LOGGER = logging.getLogger(__name__)

_SUPPORTED_ARCHITECTURES = ["opt"]


class KeyValueCacheInjector(BaseExporter):
    def __init__(
        self,
        model_path: str,
        inplace: bool = True,
    ):
        """
        A transformation that injects Key Value cache support into the model.
        This means that the autoregressive model takes as an input / returns
        as an output a cache of key value pairs that are used to speed up the
        autoregressive generation process (reduce the compute of key/value pairs
        by storing the results of previous computations in memory).

        The exporter will look for a `config.json` file in the `model_path` directory
        to determine the static dimensions of the kv cache input/output.

        This transformation not only injects the cache support, but also adjusts
        the model to account for the cache support. This means altering the input
        to the model, such as adding "position" input to the model.

        Usage:
        ```python
        onnx_model: onnx.ModelProto = ...
        exporter = KeyValueCacheInjector(model_path="path/to/model")
        exporter.export(onnx_model, "model.onnx")
        ```

        You can also just optimize the model directly without saving to disk:
        ```python
        onnx_model: onnx.ModelProto = ...
        exporter = KeyValueCacheInjector(model_path="path/to/model")
        optimized_model = exporter.apply(onnx_model)
        ```

        :param model_path: The path to the directory containing the model.
        :param inplace: If True, the model will be modified in place.
            If False, a copy of the model will be made and modified.
        """
        self.inplace = inplace
        self.config = self.get_config(model_path)

        if not self.config["model_type"] in _SUPPORTED_ARCHITECTURES:
            _LOGGER.warn(
                f"Model type {self.config.model_type} is currently not supported. "
                f"Supported model types: {_SUPPORTED_ARCHITECTURES}."
                f"Proceeding with transformation, but may require additional "
                f"customization..."
            )

        num_attention_heads = self.config["num_attention_heads"]
        hidden_size_kv_cache = self.config["hidden_size"] // num_attention_heads

        transforms = [
            CacheKeysAndValues(
                num_attention_heads=num_attention_heads,
                hidden_size_kv_cache=hidden_size_kv_cache,
            ),
            PositionEmbeddingsAdjustment(),
        ]

        super().__init__(transforms)

    def get_config(self, model_path: Union[str, Path]) -> Dict[str, Any]:
        """
        From the model path, get the config.json file and return it as a dict.

        :param model_path: The path to the directory containing the model.
        :return: The config.json file as a dict.
        """
        model_path = Path(model_path) if isinstance(model_path, str) else model_path

        if not model_path.is_dir():
            raise ValueError(
                f"`model_path` is expected to be a directory, found {model_path}"
            )
        config_file = [
            file for file in model_path.iterdir() if file.name == "config.json"
        ]
        config_file = config_file[0]

        _LOGGER.warn(f"Found config file {config_file}")

        with open(config_file) as f:
            config = json.load(f)

        return config

    def pre_validate(self, model: Union[onnx.ModelProto, str, Path]) -> onnx.ModelProto:
        if isinstance(model, (str, Path)):
            model = onnx.load(str(model))

        if not isinstance(model, onnx.ModelProto):
            raise TypeError(f"Expected onnx.ModelProto, found {type(model)}")
        return model if self.inplace else deepcopy(model)

    def post_validate(self, model: onnx.ModelProto) -> onnx.ModelProto:
        if not isinstance(model, onnx.ModelProto):
            raise TypeError(f"Expected onnx.ModelProto, found {type(model)}")
        return model

    def export(self, pre_transforms_model: onnx.ModelProto, file_path: str):
        post_transforms_model: onnx.ModelProto = self.apply(pre_transforms_model)
        save_onnx(post_transforms_model, file_path)
