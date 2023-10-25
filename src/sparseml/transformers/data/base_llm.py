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

import random
from typing import Optional

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from sparsezoo.utils.registry import RegistryMixin


class TransformersDataset(RegistryMixin, Dataset):
    def __init__(
        self,
        model: str,
        seqlen: int,
        nsamples: int,
        path: str,
        name: Optional[str] = None,
        seed: int = 0,
        split: str = "train",
        use_max_tokens: bool = True,
        split_percent_to_use: float = 1.0,
        shuffle: bool = True,
        **kwargs,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        self._nsamples = nsamples
        self._seqlen = seqlen
        self._use_max_tokens = use_max_tokens
        self._split_from_end = False
        try:
            dataset = load_dataset(path, name, **kwargs, split=split)
        except ValueError:
            dataset = load_dataset(path, name, **kwargs, split="train")
            self._split_from_end = True

        random.seed(seed)
        data = list(dataset)
        data_to_use = int(split_percent_to_use * len(data))
        self._data = data[-data_to_use:] if self._split_from_end else data[:data_to_use]
        if not self._nsamples:
            self._nsamples = len(dataset)
        if shuffle:
            random.shuffle(self._data)
        self._data = self._data[: self._nsamples]

    def create_dataloader(self, data, join_on=None):
        self.loader = []
        if self._use_max_tokens:
            data_idx = 0
            encoder = self.tokenizer(join_on.join(data), return_tensors="pt")[
                "input_ids"
            ][0]
            while self._nsamples is None or len(self.loader) < self._nsamples:
                start_idx = data_idx * self._seqlen
                end_idx = start_idx + self._seqlen
                if start_idx >= len(encoder):
                    break
                elif end_idx >= len(encoder):
                    sequence = encoder[start_idx:]
                else:
                    sequence = encoder[start_idx:end_idx]
                data_idx += 1

                tokenized_sample = self._add_end_token(sequence)
                tokenized_sample = torch.unsqueeze(tokenized_sample, dim=0)
                self.loader.append(tokenized_sample)
                if data_idx >= len(data):
                    break
        else:
            for sample in data:
                tokenized_sample = self.tokenizer(
                    sample,
                    truncation=True,
                    max_length=self._seqlen,
                    return_tensors="pt",
                    padding=False,
                )["input_ids"][0]
                tokenized_sample = self._add_end_token(tokenized_sample)
                tokenized_sample = torch.unsqueeze(tokenized_sample, dim=0)
                self.loader.append(tokenized_sample)

    def _add_end_token(self, tokenized_sample):
        if tokenized_sample[-1] != self.tokenizer.eos_token_id:
            if len(tokenized_sample) == self._seqlen:
                tokenized_sample[-1] = self.tokenizer.eos_token_id
            else:
                tokenized_sample = torch.cat(
                    (
                        tokenized_sample,
                        torch.tensor((self.tokenizer.eos_token_id,)),
                    ),
                )

        return tokenized_sample

    def __len__(self):
        return len(self.loader)

    def __item__(self, idx):
        return self.loader[idx]
