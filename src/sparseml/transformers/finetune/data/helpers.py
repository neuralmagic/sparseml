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

from datasets import Dataset, load_dataset


def get_raw_dataset(data_args, cache_dir: str, **kwargs) -> Dataset:
    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=cache_dir,
        **kwargs,
    )

    return raw_datasets

def make_dataset_splits(tokenized_datasets, do_train, do_eval, do_predict):
    if "all" in tokenized_datasets and len(tokenized_datasets) == 1:
        tokenized_datasets = tokenized_datasets["all"]
        
    train_split = eval_split = predict_split = None
    if do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_split = tokenized_datasets["train"]
    if do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_split = tokenized_datasets["validation"]
    if do_predict:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_split = tokenized_datasets["test"]

    split_datasets = {
        "train": train_split,
        "validation": eval_split,
        "test": predict_split,
    }
    return split_datasets
