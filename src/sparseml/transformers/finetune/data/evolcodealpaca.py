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
from copy import deepcopy
from typing import Optional

from sparseml.transformers.finetune.data import TextGenerationDataset
from sparseml.transformers.finetune.data.data_helpers import get_raw_dataset


@TextGenerationDataset.register(name="evolcodealpaca")
class EvolCodeAlpacaDataset(TextGenerationDataset):
    """
    Child text generation class for the Evol Code Alpaca dataset

    :param data_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `test` or `train[:5%]`
    :param tokenizer: tokenizer to use on dataset
    """

    EVOL_ALPACA_TEMPLATE = (
        "Below is an instruction that describes a "
        "programming task. Write a program that appropriately "
        "completes the request.\n\n### Instruction:\n{instruction}"
        "\n\n### Response:\n"
    )

    def __init__(self, data_args, split, tokenizer):
        data_args = deepcopy(data_args)
        data_args.dataset_name = "theblackcat102/evol-codealpaca-v1"
        super().__init__(
            text_column="text", data_args=data_args, split=split, tokenizer=tokenizer
        )

    def get_raw_dataset(self, cache_dir: Optional[str] = None):
        """
        Load the raw dataset from Hugging Face, using cached copy if available.
        Additionally reformats the entries to fit the alpaca template.

        :param cache_dir: disk location to search for cached dataset
        :return: the requested dataset
        """
        raw_dataset = get_raw_dataset(
            self.data_args,
            cache_dir,
            split=self.split,
            streaming=self.data_args.streaming,
            **self.raw_kwargs,
        )

        # helper fn for restructuring each dataset entry using the alpaca template
        def restructure_fn(sample):
            sample["text"] = self.EVOL_ALPACA_TEMPLATE.format(
                instruction=sample["instruction"]
            )
            if "output" in sample:
                sample["text"] += sample["output"]
            return sample

        raw_dataset = raw_dataset.map(
            restructure_fn,
            batched=False,
            remove_columns=["output", "instruction"],
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc="Restructuring Evol Code Alpaca Dataset",
        )
        return raw_dataset
