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

from datasets.dataset_dict import DatasetDict

from sparseml.transformers.finetune.data import TextGenerationDataset


@TextGenerationDataset.register(name="custom", alias=["json", "csv"])
class CustomDataset(TextGenerationDataset):
    """
    Child text generation class for custom local dataset supporting load
    for csv and json

    :param data_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `test` or `train[:5%]`
        Can also be set to None to load all the splits
    :param tokenizer: tokenizer to use on dataset

    """

    def __init__(self, data_args, split, tokenizer):
        data_args = deepcopy(data_args)
        super().__init__(
            text_column=data_args.text_column,
            data_args=data_args,
            split=split,
            tokenizer=tokenizer,
        )
        self.preprocessing_func = data_args.preprocessing_func
        self.remove_columns = data_args.remove_columns

    def get_raw_dataset(self, *_ignore, **__ignore) -> DatasetDict:
        """Get the raw dataset and apply preprocessing func if provided"""

        raw_dataset: DatasetDict = super().get_raw_dataset()

        if self.preprocessing_func is not None:
            raw_dataset = self.map(
                raw_dataset,
                function=self.preprocessing_func,
                batched=False,
                remove_columns=self.remove_columns,
                num_proc=self.data_args.preprocessing_num_workers,
                desc="Applying custom func to the custom dataset",
            )

        return raw_dataset
