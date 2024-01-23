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
from datasets import load_dataset


@TextGenerationDataset.register(name=["json", "csv"])
class CustomDataset(TextGenerationDataset):
    """
    Child text generation class for custom local dataset supporting load
    for csv and json

    :param data_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `test` or `train[:5%]`
    :param tokenizer: tokenizer to use on dataset
    """

    def __init__(self, data_args, split, tokenizer):
        data_args = deepcopy(data_args)
        # breakpoint()
        super().__init__(
            text_column="text", data_args=data_args, split=split, tokenizer=tokenizer
        )
        
    # def get_raw_dataset(self, cache_dir: str):
    #     """
    #     Load the dataset from the local path 

    #     :param cache_dir: disk location to search for cached dataset
    #     :return: the requested dataset
    #     """
        
    #     load_dataset("json", data_files="scratch/data.json")
    #     dataset = load_dataset(cache_dir)
        
        


    # def get_raw_dataset(self, cache_dir: Optional[str] = None):
    #     """
    #     Load the raw dataset from Hugging Face, using cached copy if available.
    #     Additionally reformats the entries to fit the alpaca template.

    #     :param cache_dir: disk location to search for cached dataset
    #     :return: the requested dataset
    #     """
    #     ## load from hf
    #     raw_dataset = super().get_raw_dataset(cache_dir=cache_dir)
    #     ##
        
    #     breakpoint()
    #     # helper fn for restructuring each dataset entry using the alpaca template
    #     def restructure_fn(sample):
    #         if "input" in sample:
    #             sample["text"] = self.ALPACA_TEMPLATE["prompt_input"].format(
    #                 instruction=sample["instruction"], input=sample["input"]
    #             )
    #         else:
    #             sample["text"] = self.ALPACA_TEMPLATE["prompt_no_input"].format(
    #                 instruction=sample["instruction"]
    #             )

    #         if "output" in sample:
    #             sample["text"] += sample["output"]
    #         return sample

    #     raw_dataset = self.map(
    #         raw_dataset,
    #         function=restructure_fn,
    #         batched=False,
    #         remove_columns=["input", "output", "instruction", "data_source"],
    #         num_proc=self.data_args.preprocessing_num_workers,
    #         load_from_cache_file=not self.data_args.overwrite_cache,
    #         desc="Restructuring Platypus Dataset",
    #     )
    #     breakpoint()
    #     return raw_dataset
