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
import torch
from datasets import IterableDataset, load_dataset

from sparseml.transformers.finetune.data import TextGenerationDataset
from sparseml.transformers.finetune.data.data_args import DataTrainingArguments
from sparseml.transformers.finetune.data.data_helpers import format_calibration_data
from sparseml.transformers.finetune.model_args import ModelArguments
from sparseml.transformers.finetune.runner import StageRunner
from sparseml.transformers.finetune.training_args import TrainingArguments


@pytest.mark.usefixtures("tiny_llama_tokenizer")
def test_concatenation_tokenization(tiny_llama_tokenizer):
    data_args = DataTrainingArguments(
        dataset="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        concatenate_data=True,
    )
    wiki_manager = TextGenerationDataset.load_from_registry(
        data_args.dataset,
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
    data_args = DataTrainingArguments(dataset="open_platypus", pad_to_max_length=False)
    op_manager = TextGenerationDataset.load_from_registry(
        data_args.dataset,
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
    data_args = DataTrainingArguments(dataset="open_platypus", max_seq_length=4096)
    op_manager = TextGenerationDataset.load_from_registry(
        data_args.dataset,
        data_args=data_args,
        split="train[80%:]",
        tokenizer=tiny_llama_tokenizer,
    )

    assert op_manager.max_seq_length == tiny_llama_tokenizer.model_max_length


# test loading percentages works as expected size-wise
@pytest.mark.usefixtures("tiny_llama_tokenizer")
def test_dataset_kwargs_and_percentages(tiny_llama_tokenizer):
    data_args = DataTrainingArguments(
        dataset="wikitext",
        raw_kwargs={
            "data_files": {"train": "wikitext-2-raw-v1/train-00000-of-00001.parquet"}
        },
    )
    c4_manager_a = TextGenerationDataset.load_from_registry(
        data_args.dataset,
        data_args=data_args,
        split="train[5%:10%]",
        tokenizer=tiny_llama_tokenizer,
    )
    raw_dataset_a = c4_manager_a.get_raw_dataset()

    c4_manager_b = TextGenerationDataset.load_from_registry(
        data_args.dataset,
        data_args=data_args,
        split="train[5%:15%]",
        tokenizer=tiny_llama_tokenizer,
    )
    raw_dataset_b = c4_manager_b.get_raw_dataset()

    assert len(raw_dataset_b) == 2 * len(raw_dataset_a)


@pytest.mark.usefixtures("tiny_llama_tokenizer")
@pytest.mark.parametrize(
    "dataset_key,dataset_config,split,do_concat",
    [
        ("ptb", "penn_treebank", "train[:5%]", False),
        ("gsm8k", "main", "train[:5%]", True),
        ("ultrachat_200k", "default", "train_sft[:2%]", False),
    ],
)
def test_datasets(tiny_llama_tokenizer, dataset_key, dataset_config, split, do_concat):
    data_args = DataTrainingArguments(
        dataset=dataset_key,
        dataset_config_name=dataset_config,
        concatenate_data=do_concat,
    )
    manager = TextGenerationDataset.load_from_registry(
        data_args.dataset,
        data_args=data_args,
        split=split,
        tokenizer=tiny_llama_tokenizer,
    )
    raw_dataset = manager.get_raw_dataset()
    assert len(raw_dataset) > 0
    assert raw_dataset.split == split
    assert raw_dataset.info.config_name == dataset_config

    tokenized_dataset = manager.tokenize_and_process(raw_dataset)
    assert "input_ids" in tokenized_dataset.features
    assert "labels" in tokenized_dataset.features
    for i in range(len(tokenized_dataset)):
        if do_concat:
            assert len(tokenized_dataset[i]["input_ids"]) == manager.max_seq_length
        else:
            assert len(tokenized_dataset[i]["input_ids"]) <= manager.max_seq_length


@pytest.mark.skip("Dataset load broken on Hugging Face")
@pytest.mark.usefixtures("tiny_llama_tokenizer")
def test_evol(tiny_llama_tokenizer):
    data_args = DataTrainingArguments(
        dataset="evolcodealpaca",
        dataset_config_name=None,
        concatenate_data=False,
    )
    evol_manager = TextGenerationDataset.load_from_registry(
        data_args.dataset,
        data_args=data_args,
        split="train[:2%]",
        tokenizer=tiny_llama_tokenizer,
    )
    raw_dataset = evol_manager.get_raw_dataset()
    assert len(raw_dataset) > 0
    assert raw_dataset.split == "train[:2%]"

    tokenized_dataset = evol_manager.tokenize_and_process(raw_dataset)
    assert "input_ids" in tokenized_dataset.features
    assert "labels" in tokenized_dataset.features
    for i in range(len(tokenized_dataset)):
        assert len(tokenized_dataset[i]["input_ids"]) <= evol_manager.max_seq_length


@pytest.mark.usefixtures("tiny_llama_tokenizer")
def test_dvc_dataloading(tiny_llama_tokenizer):
    data_args = DataTrainingArguments(
        dataset="csv",
        dataset_path="dvc://workshop/satellite-data/jan_train.csv",
        dvc_data_repository="https://github.com/iterative/dataset-registry.git",
    )
    manager = TextGenerationDataset(
        text_column="",
        data_args=data_args,
        split="train",
        tokenizer=tiny_llama_tokenizer,
    )

    raw_dataset = manager.get_raw_dataset()
    assert len(raw_dataset) > 0
    assert isinstance(raw_dataset[0], dict)


@pytest.mark.usefixtures("tiny_llama_tokenizer")
def test_stream_loading(tiny_llama_tokenizer):
    data_args = DataTrainingArguments(
        dataset="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        concatenate_data=True,
        streaming=True,
    )
    manager = TextGenerationDataset.load_from_registry(
        data_args.dataset,
        data_args=data_args,
        split="train",
        tokenizer=tiny_llama_tokenizer,
    )

    raw_dataset = manager.get_raw_dataset()
    processed = manager.tokenize_and_process(raw_dataset)
    assert isinstance(processed, IterableDataset)
    with pytest.raises(TypeError):
        # in streaming mode we don't know the length of the dataset
        _ = len(processed)

    # confirm tokenization of streamed item works correctly
    item = next(iter(processed))
    assert "labels" in item
    assert len(item["input_ids"]) == manager.max_seq_length


@pytest.mark.usefixtures("tiny_llama_tokenizer")
@pytest.mark.parametrize(
    "split_def", [("train"), ("train[60%:]"), ({"train": "train[:20%]"}), (None)]
)
def test_split_loading(split_def, tiny_llama_tokenizer):
    data_args = DataTrainingArguments(dataset="open_platypus", splits=split_def)
    training_args = TrainingArguments(do_train=True, output_dir="dummy")
    model_args = ModelArguments(model=None)
    stage_runner = StageRunner(
        model_args=model_args, data_args=data_args, training_args=training_args
    )
    stage_runner.populate_datasets(tokenizer=tiny_llama_tokenizer)

    train_dataset = stage_runner.get_dataset_split("train")
    assert train_dataset is not None
    assert isinstance(train_dataset[0], dict)


def test_load_tokenized_data(tiny_llama_tokenizer):
    dataset = load_dataset("garage-bAInd/Open-Platypus")["train"]
    NUM_CALIB_SAMPS = 256
    MAX_SEQ_LEN = 512
    dataset = dataset.shuffle(seed=42).select(range(NUM_CALIB_SAMPS))

    def preprocess(sample):
        concat_text = "INPUT: " + sample.get("input", "")
        concat_text += "INSTRUCTIONS: " + sample.get("instruction", "")
        concat_text += "OUTPUT: " + sample.get("output", "")

        return tiny_llama_tokenizer(
            concat_text, padding=False, max_length=MAX_SEQ_LEN, truncation=True
        )

    tokenized_dataset = dataset.map(
        preprocess, remove_columns=["input", "output", "instruction", "data_source"]
    )
    stage_runner = StageRunner(
        model_args=None,
        data_args=DataTrainingArguments(
            dataset=tokenized_dataset, shuffle_calibration_samples=False
        ),
        training_args=TrainingArguments(do_oneshot=True),
    )
    stage_runner.populate_datasets(tokenizer=None)
    calib_dataset = stage_runner.get_dataset_split("calibration")
    assert len(calib_dataset) == NUM_CALIB_SAMPS
    data_cols = calib_dataset.column_names
    assert len(data_cols) == 2
    assert "input_ids" in data_cols and "attention_mask" in data_cols

    # confirm turning shuffle off works
    calib_dataloader = format_calibration_data(
        tokenized_dataset=calib_dataset,
        num_calibration_samples=NUM_CALIB_SAMPS,
        do_shuffle=stage_runner._data_args.shuffle_calibration_samples,
    )
    assert len(calib_dataloader) == NUM_CALIB_SAMPS
    dataloader_sample = next(iter(calib_dataloader))["input_ids"]
    diff = dataloader_sample - torch.Tensor(calib_dataset[0]["input_ids"])
    assert torch.sum(diff) == 0
