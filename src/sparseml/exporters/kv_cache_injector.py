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
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import onnx

from sparseml.exporters.base_exporter import BaseExporter
from sparseml.exporters.transforms import OnnxTransform
from sparseml.exporters.transforms.kv_cache import (
    CacheKeysAndValues,
    KeyValueCacheConfig,
    get_kv_cache_config,
)
from sparsezoo.utils import save_onnx


_LOGGER = logging.getLogger(__name__)


class KeyValueCacheInjector(BaseExporter):
    def __init__(
        self,
        model_path: Optional[str] = None,
        inplace: bool = True,
        **kwargs: Any,
    ):
        """
        A transformation that injects Key Value cache support into the model.
        This means that the
        - autoregressive model that
            * takes input_ids and attention_mask as INPUT
            * returns logits as OUTPUT
        - is transformed into a model that
            * takes input_ids, attention_mask, and kv_cache as INPUT
            * returns logits and updated kv_cache as OUTPUT

        The goal of the KV cache injection is speed up the autoregressive
        generation process (reduce the compute of key/value pairs by storing
        the results of previous computations in memory).

        The exporter will look for a `config.json` file in the `model_path`
        directory to determine the parameters for KV cache injection.
        If `model_path` is not provided, the requested parameters can be
        provided in the `kwargs`.

        This transformation not only solely injects the kv cache
        inputs/outputs, but also adjusts the original ONNX graph to
        account for the necessary changes. This is done by the
        optional `additional_transforms` variable.

        Usage:
        ```python
        onnx_model: onnx.ModelProto = ...
        exporter = KeyValueCacheInjector(model_path="path/to/model")
        exporter.export(onnx_model, "model.onnx")
        ```

        Alternatively:
        ```python
        onnx_model: onnx.ModelProto = ...
        exporter = KeyValueCacheInjector(model_path="path/to/model",
                                         num_attention_heads = 16,
                                         hidden_size_dim = 64)
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
        :param kwargs: (Optionally) the parameters for the KV cache injection
            if no `model_path` is provided.
        """
        self.inplace = inplace

        config = get_kv_cache_config(model_path)

        if config is not None:
            transforms = self._get_transforms_from_config(config)

        elif kwargs:
            transforms = self._get_transforms_from_kwargs(kwargs)

        else:
            raise ValueError(
                f"Unable to find KeyValueCacheConfig for model_path='{model_path}'. "
                "Either kwargs must be provided to KeyValueCacheInjector to construct "
                "OnnxTransform, or a new config should be registered in "
                "`sparseml/src/sparseml/exporters/transforms/kv_cache/configs.py`"
            )

        super().__init__(transforms)

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

    @staticmethod
    def _get_transforms_from_config(config: KeyValueCacheConfig) -> List[OnnxTransform]:
        additional_transforms = config.additional_transforms

        transforms = [
            CacheKeysAndValues(
                num_attention_heads=config.num_attention_heads,
                hidden_size_kv_cache=config.hidden_size_kv_cache,
                multiply_batch_by_num_att_heads=config.multiply_batch_by_num_att_heads,
                transpose_value_input=config.transpose_value_input,
                transpose_key_input=config.transpose_key_input,
            )
        ]
        if additional_transforms is not None:
            if not isinstance(additional_transforms, list):
                additional_transforms = [additional_transforms]
            transforms += [transform() for transform in additional_transforms]

        return transforms

    @staticmethod
    def _get_transforms_from_kwargs(kwargs: Dict[str, Any]) -> List[OnnxTransform]:
        transforms = [
            CacheKeysAndValues(
                num_attention_heads=kwargs.get("num_attention_heads"),
                hidden_size_kv_cache=kwargs.get("hidden_size_kv_cache"),
                multiply_batch_by_num_att_heads=kwargs.get(
                    "multiply_batch_by_num_att_heads", False
                ),
                transpose_value_input=kwargs.get("transpose_value_input", None),
                transpose_key_input=kwargs.get("transpose_key_input", None),
            )
        ]
        return transforms
