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
from typing import Optional, Union

from datasets import Dataset, IterableDataset
from transformers import AutoTokenizer

from sparseml.modifiers.utils.pytorch_helpers import PADDING_MASK_COLUMN_NAME
from sparseml.transformers.finetune.data.data_args import DataTrainingArguments
from sparseml.transformers.finetune.data.data_helpers import (
    get_custom_datasets_from_path,
    get_raw_dataset,
)
from sparsezoo.utils.registry import RegistryMixin


_LOGGER: logging.Logger = logging.getLogger(__name__)


class TextGenerationDataset(RegistryMixin):
    """
    Base class for text datasets, handles tokenization and dataset splits

    :param text_column: name of column corresponding to text in the dataset
    :param data_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `test` or `train[:5%]`
    :param tokenizer: tokenizer to use on dataset
    """

    def __init__(
        self,
        text_column: str,
        data_args: DataTrainingArguments,
        split: str,
        tokenizer: AutoTokenizer,
    ):
        self.text_column = text_column
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.raw_kwargs = data_args.raw_kwargs or {}
        self.split = split
        self.dvc_dataset = (
            True if self.data_args.dvc_data_repository is not None else False
        )
        self.custom_dataset = True if self.data_args.dataset_path is not None else False

        # configure padding
        if data_args.concatenate_data:
            self.padding = False
        elif data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            self.padding = False

        if self.padding and self.tokenizer:
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # configure sequence length
        max_seq_length = data_args.max_seq_length
        model_max_length = tokenizer.model_max_length if tokenizer else max_seq_length
        if self.tokenizer and max_seq_length > model_max_length:
            _LOGGER.warning(
                f"The max_seq_length passed ({max_seq_length}) is larger than "
                f"the maximum length for the model ({tokenizer.model_max_length}). "
                f"Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, model_max_length)

    def get_raw_dataset(self, cache_dir: Optional[str] = None) -> Dataset:
        """
        Load the raw dataset from Hugging Face, using cached copy if available

        :param cache_dir: disk location to search for cached dataset
        :return: the requested dataset
        """
        if self.custom_dataset:
            if self.dvc_dataset:
                self.raw_kwargs["storage_options"] = {
                    "url": self.data_args.dvc_data_repository
                }
                self.raw_kwargs["data_files"] = self.data_args.dataset_path
            else:
                self.raw_kwargs["data_files"] = get_custom_datasets_from_path(
                    self.data_args.dataset_path,
                    self.data_args.dataset_name,
                )

        return get_raw_dataset(
            self.data_args,
            cache_dir,
            split=self.split,
            streaming=self.data_args.streaming,
            **self.raw_kwargs,
        )

    def tokenize_and_process(
        self, raw_dataset: Dataset, store_padding_mask: bool = False
    ) -> Dataset:
        """
        Sets up the raw dataset for finetuning, performs tokenization, concatenates
        entries to max sequence length if desired, and adds labels to each entry

        :param raw_dataset: dataset to process
        :param store_padding_mask: when set, keep track of a padding mask for each
        embedding in the dataset. Used for zeroing out padding during one-shot pruning.
        """
        # helper fn for tokenizing text column
        def tokenize_fn(data):
            result = self.tokenizer(
                data[self.text_column],
                padding=self.padding,
                max_length=self.max_seq_length,
                truncation=True,
            )
            return result

        # helper fn for filling to max_sequence_length by concatenating entries
        def group_text_fn(data):
            concatenated_data = {k: sum(data[k], []) for k in data.keys()}
            total_length = len(concatenated_data[list(data.keys())[0]])
            total_length = (total_length // self.max_seq_length) * self.max_seq_length
            result = {
                k: [
                    t[i : i + self.max_seq_length]
                    for i in range(0, total_length, self.max_seq_length)
                ]
                for k, t in concatenated_data.items()
            }
            return result

        # helper fn for adding labels, needed for loss calculation
        def label_fn(data):
            data["labels"] = data["input_ids"].copy()
            return data

        dataset = self.map(
            raw_dataset,
            function=tokenize_fn,
            batched=True,
            remove_columns=[self.text_column] if not store_padding_mask else None,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

        if store_padding_mask:
            dataset = self.create_padding_mask(dataset)

        if self.data_args.concatenate_data:
            dataset = self.map(
                dataset,
                function=group_text_fn,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Grouping text",
            )

        dataset = self.map(
            dataset,
            function=label_fn,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc="Adding labels",
        )

        return dataset

    def create_padding_mask(self, raw_dataset: Dataset) -> Dataset:
        """
        Given a dataset, add a new column to each entry that is a mask indicating
        whether or not that element is padding

        :param raw_dataset: dataset to add padding column to
        :return: dataset with padding column added
        """
        # helper fn for tokenizing text column
        def padding_mask_fn(data):
            result = self.tokenizer(
                data[self.text_column],
                padding=False,
                max_length=self.max_seq_length,
                truncation=True,
            )
            non_padded_size = len(result["input_ids"])
            mask = [1] * non_padded_size
            padding = [0] * (self.max_seq_length - non_padded_size)
            data[PADDING_MASK_COLUMN_NAME] = mask + padding
            return data

        padding_mask_dataset = self.map(
            raw_dataset,
            function=padding_mask_fn,
            remove_columns=[self.text_column],
            batched=False,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc="Creating padding mask",
        )

        return padding_mask_dataset

    def map(
        self, dataset: Union[Dataset, IterableDataset], **kwargs
    ) -> Union[Dataset, IterableDataset]:
        """
        Wrapper function around Dataset.map and IterableDataset.map, clears invalid
        parameters in the case where streaming is enabled

        :param dataset: dataset to apply mapping to
        :param kwargs: args to pass on to map function
        :return: mapped dataset
        """
        if isinstance(dataset, IterableDataset):
            # remove arguments that don't apply to streaming
            kwargs.pop("num_proc", None)
            kwargs.pop("load_from_cache_file", None)
            kwargs.pop("desc", None)

        return dataset.map(**kwargs)
