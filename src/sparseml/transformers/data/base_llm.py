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
from torch.nn import Module
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from sparsezoo.utils.registry import RegistryMixin


class TransformersDataset(RegistryMixin, Dataset):
    def __init__(
        self,
        model: Module,
        seqlen: int,
        nsamples: int,
        path: str,
        name: Optional[str] = None,
        seed: int = 0,
        split: str = "train",
        **kwargs,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        self._nsamples = nsamples
        self._seqlen = seqlen
        dataset = load_dataset(path, name, **kwargs, split=split)
        if not self._nsamples:
            self._nsamples = len(dataset)

        random.seed(seed)
        data = list(dataset)
        random.shuffle(data)
        self._data = data[:nsamples]

    def create_dataloader(self, data):
        self.loader = []
        for sample in data:
            tokenized_sample = self.tokenizer(
                sample,
                truncation=True,
                max_length=self._seqlen,
                return_tensors="pt",
                padding=False,
            )["input_ids"][0]

            if tokenized_sample[-1] != self.tokenizer.eos_token_id:
                if len(tokenized_sample) == self._seqlen:
                    tokenized_sample[-1] = self.tokenizer.eos_token_id
                else:
                    tokenized_sample = torch.concatenate(
                        (
                            tokenized_sample,
                            torch.tensor((self.tokenizer.eos_token_id,)),
                        ),
                    )

            tokenized_sample = torch.unsqueeze(tokenized_sample, dim=0)
            self.loader.append(tokenized_sample)

    def __len__(self):
        return self._nsamples

    def __item__(self, idx):
        return self.loader[idx]
