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

from datasets import Dataset, load_dataset


__all__ = ["get_raw_dataset", "make_dataset_splits"]


def get_raw_dataset(data_args, cache_dir: Optional[str] = None, **kwargs) -> Dataset:
    """
    Load the raw dataset from Hugging Face, using cached copy if available

    :param cache_dir: disk location to search for cached dataset
    :return: the requested dataset
    """
    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=cache_dir,
        **kwargs,
    )

    return raw_datasets


def make_dataset_splits(
    tokenized_datasets: Dict[str, Any], do_train: bool, do_eval: bool, do_predict: bool
) -> Dict[str, Dataset]:
    """
    Restructures the datasets dictionary based on what tasks will be run
    (train, eval, predict)

    :param tokenized_datasets: dictionary of processed datasets
    :param do_train: Whether to store the train dataset
    :param do_eval: Whether to store the validation dataset
    :param do_predict: Whether to store the test dataset
    :return: Datasets to be used by the requested tasks
    """

    # handles case where all splits are contained in a single dataset
    if "all" in tokenized_datasets and len(tokenized_datasets) == 1:
        tokenized_datasets = tokenized_datasets.get("all")

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
        if "test" not in tokenized_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_split = tokenized_datasets["test"]

    split_datasets = {
        "train": train_split,
        "validation": eval_split,
        "test": predict_split,
    }
    return split_datasets
