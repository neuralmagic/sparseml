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
from typing import Dict

from datasets import load_dataset


ALPACA_TEMPLATE = {
    "prompt_input": "Below is an instruction that describes a task, paired with an "
    "input that provides further context. Write a response that appropriately "
    "completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n"
    "{input}\n\n### Response:\n",
    "prompt_no_input": "Below is an instruction that describes a task. Write a "
    "response that appropriately completes the request.\n\n### Instruction:\n{"
    "instruction}\n\n### Response:\n",
}

GSM_TEMPLATE = "Question: {question}.\nAnswer: "


def _fetch_open_platypus_dataset():
    dataset = load_dataset("garage-bAInd/Open-Platypus")["train"]
    dataset = dataset.shuffle(seed=42).select(range(256))
    return dataset


def _fetch_gsm8k_data():
    dataset = load_dataset("gsm8k", "main")["train"]
    dataset = dataset.shuffle(seed=42).select(range(256))
    return dataset


def _preprocess_alpaca(sample):
    if "input" in sample:
        concat_text = ALPACA_TEMPLATE["prompt_input"].format(
            instruction=sample["instruction"], input=sample["input"]
        )
    else:
        concat_text = ALPACA_TEMPLATE["prompt_no_input"].format(
            instruction=sample["instruction"]
        )
    if "output" in sample:
        concat_text += sample["output"]

    return concat_text


def _preprocess_gsm(sample):
    concat_text = GSM_TEMPLATE.format(question=sample["question"])
    concat_text += sample["answer"]
    return concat_text


def get_data_utils(dataset_name: str) -> Dict:
    """
    Given the name of a dataset, fetch the appropriate set of data processing utils.
    Returns a dictionary of data processing utils required to process the data when
    providing tokenized data to oneshot.
    Includes:
        1. dataload: function to load the dataset
        2. preprocess: preprocessing function to apply to the dataset
        3. remove_columns: specific columns which should be removed from the dataset

    :param dataset_name: the name of the dataset
    :returns dictionary of preprocessing functions/utils.
    """
    data_mapping = {
        "open_platypus": {
            "preprocess": _preprocess_alpaca,
            "dataload": _fetch_open_platypus_dataset,
            "remove_columns": ["input", "output", "instruction", "data_source"],
        },
        "gsm8k": {
            "preprocess": _preprocess_gsm,
            "dataload": _fetch_gsm8k_data,
            "remove_columns": ["question", "answer"],
        },
    }
    return data_mapping.get(dataset_name)
