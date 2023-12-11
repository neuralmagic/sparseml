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

from typing import Any

from torch.nn import Module

from sparseml.transformers.utils.helpers import TaskNames
from sparseml.transformers.utils.sparse_auto_model import SparseAutoModel


def load_task_model(
    task: str, model_path: str, config: Any, trust_remote_code: bool = False
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
        return SparseAutoModel.text_generation_from_pretrained(
            model_name_or_path=model_path,
            config=config,
            model_type="model",
            trust_remote_code=trust_remote_code,
        )

    raise ValueError(f"unrecognized task given of {task}")
