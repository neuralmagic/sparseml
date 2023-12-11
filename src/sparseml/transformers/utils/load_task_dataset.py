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

from typing import Any, Dict, Optional

from torch.nn import Module
from transformers import AutoConfig, AutoTokenizer

from sparseml.transformers.utils.helpers import TaskNames


def load_task_dataset(
    task: str,
    tokenizer: AutoTokenizer,
    data_args: Dict[str, Any],
    model: Module,
    config: Optional[AutoConfig] = None,
):
    """

    Load a dataset for a given task.

    :param task: the task a dataset being loaded for
    :param tokenizer: the tokenizer to use for the dataset
    :param data_args: additional data args used to create a `DataTrainingArguments`
        instance for fetching the dataset
    :param model: the model to use for the dataset
    :param config: the config to use for the dataset
    """

    if task in TaskNames.mlm.value:
        from sparseml.transformers.masked_language_modeling import (
            DataTrainingArguments,
            get_tokenized_mlm_dataset,
        )

        data_training_args = DataTrainingArguments(**data_args)
        return get_tokenized_mlm_dataset(
            data_args=data_training_args, tokenizer=tokenizer
        )

    if task in TaskNames.qa.value:
        from sparseml.transformers.question_answering import (
            DataTrainingArguments,
            get_tokenized_qa_dataset,
        )

        data_training_args = DataTrainingArguments(**data_args)
        return get_tokenized_qa_dataset(
            data_args=data_training_args, tokenizer=tokenizer
        )

    if task in TaskNames.token_classification.value:
        from sparseml.transformers.token_classification import (
            DataTrainingArguments,
            get_tokenized_token_classification_dataset,
        )

        data_training_args = DataTrainingArguments(**data_args)
        return get_tokenized_token_classification_dataset(
            data_args=data_training_args, tokenizer=tokenizer, model=model or config
        )

    if task in TaskNames.text_classification.value:
        from sparseml.transformers.text_classification import (
            DataTrainingArguments,
            get_tokenized_text_classification_dataset,
        )

        data_training_args = DataTrainingArguments(**data_args)

        return get_tokenized_text_classification_dataset(
            data_args=data_training_args,
            tokenizer=tokenizer,
            model=model,
            config=config,
        )

    raise ValueError(f"unrecognized task given of {task}")
