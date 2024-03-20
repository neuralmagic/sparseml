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
import shutil
from pathlib import Path
from typing import Union

import onnx

from sparseml.exporters.kv_cache_injector import KeyValueCacheInjector
from sparseml.transformers.utils.helpers import ONNX_MODEL_NAME_INTERMEDIATE


__all__ = ["apply_kv_cache_injection"]

_LOGGER = logging.getLogger(__name__)


def apply_kv_cache_injection(onnx_model_path: Union[str, Path]) -> bool:
    """
    Apply key value cache injection to an ONNX model.
    Before the injection, a copy of the model is created
    at the same location under ONNX_MODEL_NAME_INTERMEDIATE.

    :param onnx_model_path: path to the ONNX model to inject
    :return: True if successful, False otherwise
    """
    create_model_copy(onnx_model_path)

    onnx_model = onnx.load(onnx_model_path, load_external_data=False)
    model_path = os.path.dirname(onnx_model_path)
    exporter = KeyValueCacheInjector(model_path=model_path)
    exporter.export(onnx_model, onnx_model_path)
    return True


def create_model_copy(
    onnx_model_path: Union[str, Path], copy_name: str = ONNX_MODEL_NAME_INTERMEDIATE
):
    copy_model_path = Path(onnx_model_path).parent / copy_name
    shutil.copyfile(src=onnx_model_path, dst=copy_model_path)
    _LOGGER.info(
        "Created a copy of the ONNX model before KV "
        f"cache injection at {copy_model_path}"
    )
