# test both cases of make dataset splits
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

import pytest

from sparseml.transformers.finetune.data import TextGenerationDataset
from sparseml.transformers.finetune.data.data_args import DataTrainingArguments


@pytest.mark.usefixtures("tiny_llama_tokenizer")
def test_concatenation_tokenization(tiny_llama_tokenizer):
    data_args = DataTrainingArguments(
        dataset_name="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        concatenate_data=True,
    )
    wiki_manager = TextGenerationDataset.load_from_registry(
        data_args.dataset_name,
        data_args=data_args,
        split="train[:5%]",
        tokenizer=tiny_llama_tokenizer,
    )
    raw_dataset = wiki_manager.get_raw_dataset()
    assert len(raw_dataset) > 0
    assert raw_dataset.split == "train[:5%]"
    assert raw_dataset.info.config_name == "wikitext-2-raw-v1"
    tokenized_dataset = wiki_manager.tokenize_and_process(raw_dataset)
    assert "input_ids" in tokenized_dataset.features
    assert "labels" in tokenized_dataset.features
    for i in range(len(tokenized_dataset)):
        assert len(tokenized_dataset[i]["input_ids"]) == wiki_manager.max_seq_length


@pytest.mark.usefixtures("tiny_llama_tokenizer")
def test_no_padding_tokenization(tiny_llama_tokenizer):
    data_args = DataTrainingArguments(
        dataset_name="open_platypus", pad_to_max_length=False
    )
    op_manager = TextGenerationDataset.load_from_registry(
        data_args.dataset_name,
        data_args=data_args,
        split="train[5%:10%]",
        tokenizer=tiny_llama_tokenizer,
    )
    raw_dataset = op_manager.get_raw_dataset()
    assert len(raw_dataset) > 0
    ex_item = raw_dataset[0]["text"]
    assert "Below is an instruction that describes a task" in ex_item

    assert raw_dataset.split == "train[5%:10%]"
    tokenized_dataset = op_manager.tokenize_and_process(raw_dataset)
    assert "input_ids" in tokenized_dataset.features
    assert "labels" in tokenized_dataset.features
    print(tokenized_dataset[0]["input_ids"])

    for i in range(len(tokenized_dataset)):
        assert len(tokenized_dataset[i]["input_ids"]) <= op_manager.max_seq_length


@pytest.mark.usefixtures("tiny_llama_tokenizer")
def test_max_seq_len_clipped(tiny_llama_tokenizer):
    data_args = DataTrainingArguments(dataset_name="open_platypus", max_seq_length=4096)
    op_manager = TextGenerationDataset.load_from_registry(
        data_args.dataset_name,
        data_args=data_args,
        split="train[80%:]",
        tokenizer=tiny_llama_tokenizer,
    )

    assert op_manager.max_seq_length == tiny_llama_tokenizer.model_max_length


# test loading percentages works as expected size-wise
@pytest.mark.usefixtures("tiny_llama_tokenizer")
def test_dataset_kwargs_and_percentages(tiny_llama_tokenizer):
    data_args = DataTrainingArguments(
        dataset_name="c4",
        dataset_config_name="allenai--c4",
        raw_kwargs={"data_files": {"train": "en/c4-train.00000-of-01024.json.gz"}},
    )
    c4_manager_a = TextGenerationDataset.load_from_registry(
        data_args.dataset_name,
        data_args=data_args,
        split="train[5%:10%]",
        tokenizer=tiny_llama_tokenizer,
    )
    raw_dataset_a = c4_manager_a.get_raw_dataset()

    c4_manager_b = TextGenerationDataset.load_from_registry(
        data_args.dataset_name,
        data_args=data_args,
        split="train[5%:15%]",
        tokenizer=tiny_llama_tokenizer,
    )
    raw_dataset_b = c4_manager_b.get_raw_dataset()

    assert len(raw_dataset_b) == 2 * len(raw_dataset_a)
