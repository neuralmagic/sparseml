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

from pathlib import Path
from typing import Any, Optional, Union

from torch.nn import Module

from sparseml.transformers.sparsification.sparse_model import SparseAutoModel
from sparseml.transformers.utils.helpers import TaskNames


__all__ = ["load_task_model"]


def load_task_model(
    task: str,
    model_path: str,
    config: Any,
    recipe: Optional[Union[str, Path]] = None,
    trust_remote_code: bool = False,
    **kwargs,
) -> Module:
    if task in TaskNames.mlm.value:
        return SparseAutoModel.masked_language_modeling_from_pretrained(
            model_name_or_path=model_path,
            config=config,
            model_type="model",
            trust_remote_code=trust_remote_code,
        )

    if task in TaskNames.qa.value:
        return SparseAutoModel.question_answering_from_pretrained(
            model_name_or_path=model_path,
            config=config,
            model_type="model",
            trust_remote_code=trust_remote_code,
        )

    if task in TaskNames.text_classification.value:
        return SparseAutoModel.text_classification_from_pretrained(
            model_name_or_path=model_path,
            config=config,
            model_type="model",
            trust_remote_code=trust_remote_code,
        )

    if task in TaskNames.token_classification.value:
        return SparseAutoModel.token_classification_from_pretrained(
            model_name_or_path=model_path,
            config=config,
            model_type="model",
            trust_remote_code=trust_remote_code,
        )

    if task in TaskNames.text_generation.value:
        torch_dtype = kwargs.get("torch_dtype")
        device_map = kwargs.get("device_map")
        sequence_length = kwargs.get("sequence_length")
        return SparseAutoModel.text_generation_from_pretrained(
            model_name_or_path=model_path,
            sequence_length=sequence_length,
            recipe=recipe,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=device_map,
            config=config,
        )

    raise ValueError(f"unrecognized task given of {task}")
