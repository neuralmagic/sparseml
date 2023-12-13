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
import os
from pathlib import Path
from typing import Union

import onnx

from sparseml.exporters.kv_cache_injector import KeyValueCacheInjector


__all__ = ["apply_kv_cache_injection"]

_LOGGER = logging.getLogger(__name__)


def apply_kv_cache_injection(onnx_model_path: Union[str, Path]):
    """
    Apply key value cache injection to an ONNX model

    :param onnx_model_path: path to the ONNX model to inject
    """
    onnx_model = onnx.load(onnx_model_path, load_external_data=False)
    model_path = os.path.dirname(onnx_model)
    exporter = KeyValueCacheInjector(model_path=model_path)
    exporter.export(onnx_model, onnx_model_path)
    _LOGGER.info(
        "Successfully applied key value cache injection "
        f"to ONNX model: {onnx_model_path}"
    )
