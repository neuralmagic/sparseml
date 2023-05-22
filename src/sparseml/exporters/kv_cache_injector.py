
from copy import deepcopy
from pathlib import Path
from typing import Union

import onnx

from sparseml.exporters.base_exporter import BaseExporter
from sparseml.exporters.transforms.kv_cache import (
    CacheKeysAndValues,
    PositionEmbeddingsAdjustment,
)
from sparsezoo import validate_onnx
from sparsezoo.utils import save_onnx


_SUPPORTED_ARCHITECTURES = ["opt"]


class KeyValueCacheInjector(BaseExporter):
    def __init__(
        self,
        model_type: str,
        inplace: bool = True,
    ):
        """
        A transformation that injects Key Value cache support into the model.
        This means that the autoregressive model takes as an input / returns
        as an output a cache of key value pairs that are used to speed up the
        autoregressive generation process (reduce the compute of key/value pairs
        by storing the results of previous computations in memory).

        This transformation not only injects the cache support, but also adjusts
        the model to account for the cache support. This means altering the input
        to the model, such as adding "position" input to the model.

        Usage:
        ```python
        onnx_model: onnx.ModelProto = ...
        exporter = KeyValueCacheInjector()
        exporter.export(onnx_model, "model.onnx")
        ```

        You can also just optimize the model directly without saving to disk:
        ```python
        onnx_model: onnx.ModelProto = ...
        exporter = KeyValueCacheInjector()
        optimized_model = exporter.apply(onnx_model)
        ```

        :param model_type: The type of model to inject the cache support into.
        :param inplace: If True, the model will be modified in place.
            If False, a copy of the model will be made and modified.
        """
        self.inplace = inplace
        if model_type.lower() not in _SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"`model_type` must be one of {_SUPPORTED_ARCHITECTURES}, "
                f"found {model_type}"
            )

        transforms = [
            CacheKeysAndValues(),
            # PositionEmbeddingAdjustment is specific for
            # OPT model, let's make it more generic in future
            PositionEmbeddingsAdjustment(),
        ]

        super().__init__(transforms)

    def pre_validate(self, model: Union[onnx.ModelProto, str, Path]) -> onnx.ModelProto:
        if isinstance(model, (str, Path)):
            validate_onnx(str(model))
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
