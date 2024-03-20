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
from typing import Any, Dict, Optional, Union

import datasets
from torch.nn import Module
from transformers import AutoConfig

from sparseml.transformers.utils.helpers import TaskNames


__all__ = ["load_task_dataset", "load_dataset"]
_LOGGER = logging.getLogger(__name__)


def load_dataset(*args, **kwargs):
    # a wrapper around datasets.load_dataset
    # to be expanded in the future
    return datasets.load_dataset(*args, **kwargs)


def load_task_dataset(
    task: str,
    tokenizer: "AutoTokenizer",  # noqa F821
    data_args: Dict[str, Any],
    model: Module,
    split: Optional[str] = None,
    config: Optional[AutoConfig] = None,
) -> Any:
    """

    Load a dataset for a given task.

    Note: datasets for task: text-generation are loaded differently than other tasks
    using the TextGenerationDataset object

    :param task: the task a dataset being loaded for
    :param tokenizer: the tokenizer to use for the dataset
    :param data_args: additional data args used to create a `DataTrainingArguments`
        instance for fetching the dataset
    :param model: the model to use for the dataset
    :param split: the split to use for the dataset.
    :param config: the config to use for the dataset
    :return: the dataset for the given task
    """
    dataset = None

    if task in TaskNames.mlm.value:
        from sparseml.transformers.masked_language_modeling import (
            DataTrainingArguments,
            get_tokenized_mlm_dataset,
        )

        data_training_args = DataTrainingArguments(**data_args)
        dataset = get_tokenized_mlm_dataset(
            data_args=data_training_args, tokenizer=tokenizer
        )

    if task in TaskNames.qa.value:
        from sparseml.transformers.question_answering import (
            DataTrainingArguments,
            get_tokenized_qa_dataset,
        )

        data_training_args = DataTrainingArguments(**data_args)
        dataset = get_tokenized_qa_dataset(
            data_args=data_training_args, tokenizer=tokenizer
        )

    if task in TaskNames.token_classification.value:
        from sparseml.transformers.token_classification import (
            DataTrainingArguments,
            get_tokenized_token_classification_dataset,
        )

        data_training_args = DataTrainingArguments(**data_args)
        dataset = get_tokenized_token_classification_dataset(
            data_args=data_training_args, tokenizer=tokenizer, model=model or config
        )

    if task in TaskNames.text_classification.value:
        from sparseml.transformers.text_classification import (
            DataTrainingArguments,
            get_tokenized_text_classification_dataset,
        )

        data_training_args = DataTrainingArguments(**data_args)

        dataset = get_tokenized_text_classification_dataset(
            data_args=data_training_args,
            tokenizer=tokenizer,
            model=model,
            config=config,
        )

    if task in TaskNames.text_generation.value:
        from sparseml.transformers.finetune.data.base import TextGenerationDataset
        from sparseml.transformers.finetune.data.data_args import DataTrainingArguments

        data_training_args = DataTrainingArguments(**data_args)
        dataset_manager = TextGenerationDataset.load_from_registry(
            data_args["dataset"],
            tokenizer=tokenizer,
            data_args=data_training_args,
            split=None,
        )
        raw_dataset = dataset_manager.get_raw_dataset()
        raw_dataset = choose_split(raw_dataset, split=split)
        dataset = dataset_manager.tokenize_and_process(raw_dataset)
        return dataset

    if dataset is None:
        raise ValueError(f"unrecognized task given of {task}")

    return choose_split(dataset, split=split)


def choose_split(
    dataset: Union[dict, "DatasetDict"], split: Optional[str] = None  # noqa F821
) -> "Dataset":  # noqa F821
    if split is None:
        _LOGGER.info(
            "Attempting to load the dataset without having specified "
            "the specific split. This may cause longer loading times for big datasets. "
            "To avoid this, the default split will be inferred..."
        )
        if "validation" in list(dataset.keys()):
            split = "validation"
        else:
            split = list(dataset.keys())[-1]
        _LOGGER.info(f"Inferred split: {split}")
    dataset = dataset[split]
    return dataset
