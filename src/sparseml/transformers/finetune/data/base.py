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

from sparseml.transformers.finetune.data.helpers import get_raw_dataset
from sparsezoo.utils.registry import RegistryMixin


_LOGGER: logging.Logger = logging.getLogger(__name__)


class TextGenerationDataset(RegistryMixin):
    def __init__(self, text_column, data_args, split, tokenizer):

        self.text_column = text_column
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.raw_kwargs = data_args.raw_kwargs or {}
        self.split = split

        if data_args.concatenate_data:
            self.padding = False
        elif data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            self.padding = False

        if self.padding:
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        max_seq_length = data_args.max_seq_length
        if max_seq_length > tokenizer.model_max_length:
            _LOGGER.warning(
                f"The max_seq_length passed ({max_seq_length}) is larger than "
                f"the maximum length for the model ({tokenizer.model_max_length}). "
                f"Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def get_raw_dataset(self, cache_dir):
        return get_raw_dataset(
            self.data_args, cache_dir, split=self.split, **self.raw_kwargs
        )

    def tokenize_and_process(self, raw_dataset):
        def tokenize_fn(data):
            result = self.tokenizer(
                data[self.text_column],
                padding=self.padding,
                max_length=self.max_seq_length,
                truncation=True,
            )
            return result

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

        def label_fn(data):
            data["labels"] = data["input_ids"].copy()
            return data

        dataset = raw_dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=[self.text_column],
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

        if self.data_args.concatenate_data:
            dataset = dataset.map(
                group_text_fn,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Grouping text",
            )

        dataset = dataset.map(
            label_fn,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc="Adding labels",
        )

        return dataset
