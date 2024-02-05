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

import os
from typing import Any, Callable, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, RandomSampler
from transformers.data import default_data_collator


__all__ = [
    "format_calibration_data",
    "get_raw_dataset",
    "make_dataset_splits",
    "get_custom_datasets_from_path",
]


def format_calibration_data(
    tokenized_dataset: Dataset,
    num_calibration_samples: Optional[int] = None,
    collate_fn: Callable = default_data_collator,
    accelerator: Optional[Any] = None,
) -> List[torch.Tensor]:
    """
    Creates a dataloader out of the calibration dataset split, trimming it to
    the desired number of calibration samples

    :param tokenized_dataset: dataset to convert to dataloader
    :param num_calibration_samples: number of data samples to convert
    :param collate_fn: optional custom collate function, or use default
    :param accelerator: optional accelerator for if preparing in FSDP mode
    :return: list of trimmed calibration data tensors
    """
    num_calibration_samples = num_calibration_samples or len(tokenized_dataset)
    shuffled_calibration = tokenized_dataset.shuffle()
    shuffled_calibration = shuffled_calibration.select(range(num_calibration_samples))

    dataloader_params = {
        "batch_size": 1,
        "sampler": RandomSampler(shuffled_calibration),
        "collate_fn": collate_fn,
        "pin_memory": True,
    }

    calib_dataloader = DataLoader(shuffled_calibration, **dataloader_params)
    if accelerator:
        calib_dataloader = accelerator.prepare(calib_dataloader)

    return calib_dataloader


def get_raw_dataset(
    data_args,
    cache_dir: Optional[str] = None,
    streaming: Optional[bool] = False,
    **kwargs,
) -> Dataset:
    """
    Load the raw dataset from Hugging Face, using cached copy if available

    :param cache_dir: disk location to search for cached dataset
    :param streaming: True to stream data from Hugging Face, otherwise download
    :return: the requested dataset

    """
    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=cache_dir,
        streaming=streaming,
        **kwargs,
    )
    return raw_datasets


def make_dataset_splits(
    tokenized_datasets: Dict[str, Any],
    do_train: bool = False,
    do_eval: bool = False,
    do_predict: bool = False,
    do_oneshot: bool = False,
) -> Dict[str, Dataset]:
    """
    Restructures the datasets dictionary based on what tasks will be run
    (train, eval, predict)

    :param tokenized_datasets: dictionary of processed datasets
    :param do_train: Whether to store the train dataset
    :param do_eval: Whether to store the validation dataset
    :param do_predict: Whether to store the test dataset
    :param do_oneshot: Whether to store the calibration dataset
    :return: Datasets to be used by the requested tasks
    """

    # handles case where all splits are contained in a single dataset
    if "all" in tokenized_datasets and len(tokenized_datasets) == 1:
        tokenized_datasets = tokenized_datasets.get("all")

    train_split = eval_split = predict_split = calib_split = None
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
    if do_oneshot:
        calib_split = tokenized_datasets.get("calibration")
        if calib_split is None:
            if "train" not in tokenized_datasets:
                raise ValueError("--do_oneshot requires a calibration dataset")
            calib_split = tokenized_datasets["train"]

    split_datasets = {
        "train": train_split,
        "validation": eval_split,
        "test": predict_split,
        "calibration": calib_split,
    }
    return split_datasets


def get_custom_datasets_from_path(path: str, ext: str = "json") -> Dict[str, str]:
    """
    Get a dictionary of custom datasets from a directory path. Support HF's load_dataset
     for local folder datasets https://huggingface.co/docs/datasets/loading

    This function scans the specified directory path for files with a
     specific extension (default is '.json').
    It constructs a dictionary where the keys are either subdirectory names or
     direct dataset names (depending on the directory structure)
    and the values are either file paths (if only one file exists with that name) or
     lists of file paths (if multiple files exist).

    :param path: The path to the directory containing the dataset files.
    :param ext: The file extension to filter files by. Default is 'json'.

    :return: A dictionary mapping dataset names to their file paths or lists of
     file paths.

    Example:
        dataset = get_custom_datasets_from_path("/path/to/dataset/directory", "json")

    Note:
        If datasets are organized in subdirectories, the function constructs the
         dictionary with lists of file paths.
        If datasets are found directly in the main directory, they are included with
         their respective names.

    Accepts:
        - path\
            train.json
            test.json
            val.json

        - path\
            train\
                data1.json
                data2.json
                ...
            test\
                ...
            val\
                ...

    """
    data_files = {}

    if any(filename.endswith(ext) for filename in os.listdir(path)):
        # If there are files with the given extension in the path
        for filename in os.listdir(path):
            if filename.endswith(ext):
                name, _ = os.path.splitext(filename)
                data_files[name] = os.path.join(path, filename)
    else:
        # If datasets are organized in subdirectories
        for root, dirs, files in os.walk(path):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                dir_dataset = []
                for filename in os.listdir(dir_path):
                    if filename.endswith(ext):
                        file_path = os.path.join(dir_path, filename)
                        dir_dataset.append(file_path)
                if dir_dataset:
                    data_files[dir_name] = dir_dataset

    return data_files
