#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/huggingface/transformers
# neuralmagic: no copyright

"""
Finetuning the library models for sequence classification on GLUE
"""

# You can also adapt this script on your own text classification task.
# Pointers for this are left as comments.

import logging
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Optional

import datasets
import transformers
from datasets import load_dataset
from sklearn.model_selection import StratifiedShuffleSplit
from torch.nn import Module
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from sparseml.pytorch.utils.distributed import record
from sparseml.transformers.sparsification import Trainer, TrainingArguments
from sparseml.transformers.utils import SparseAutoModel, get_shared_tokenizer_src


_LOGGER: logging.Logger = logging.getLogger(__name__)

metadata_args = [
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "fp16",
]


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for
    training and eval

    Using `HfArgumentParser` we can turn this class into argparse
    arguments to be able to specify them on the command line
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)"},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": ("The configuration name of the dataset to use"),
        },
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. "
            "Sequences longer  than this will be truncated, sequences shorter will "
            "be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. If False, "
            "will pad the samples dynamically when batching to the maximum length "
            "in the batch (which can be faster on GPU but will be slower on TPU)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number "
            "of training examples to this value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number "
            "of evaluation examples to this value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of "
                "prediction examples to this value if set."
            ),
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."},
    )
    validation_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "Percentage of the training data to be used as validation."},
    )
    eval_on_test: bool = field(
        default=False,
        metadata={"help": "Evaluate the test dataset."},
    )
    input_column_names: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "name of column to read model input data from. May also be comma "
                "separated list of two columns to use as inputs. Examples include "
                "'sentence' for single column and 'sentence_1,sentence_2' for two. "
                "Default behavior is to read columns based on task name or infer from "
                "non 'label' columns if sentence_column_names and task name not"
                "provided"
            )
        },
    )
    one_shot: bool = field(
        default=False,
        metadata={"help": "Whether to apply recipe in a one shot manner."},
    )
    num_export_samples: int = field(
        default=0,
        metadata={"help": "Number of samples (inputs/outputs) to export during eval."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from
    """

    model_name_or_path: str = field(
        metadata={
            "help": (
                "Path to pretrained model, sparsezoo stub. or model identifier from "
                "huggingface.co/models"
            )
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained data from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizers. Default True"},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use "
            "(can be a branch name, tag name or commit id)"
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use token generated when running `transformers-cli login` "
            "(necessary to use this script with private models)"
        },
    )


@record
def main(**kwargs):
    # See all possible arguments in
    # src/sparseml/transformers/sparsification/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_dict(kwargs)
    # Setup logging

    log_level = training_args.get_process_log_level()
    _LOGGER.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    _LOGGER.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16}"
    )
    _LOGGER.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and (len(os.listdir(training_args.output_dir)) > 0):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already "
                "exists and is not empty. Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            _LOGGER.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To "
                "avoid this behavior, change  the `--output_dir` or add "
                "`--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    raw_datasets = _get_raw_dataset(
        data_args, cache_dir=model_args.cache_dir, do_predict=training_args.do_predict
    )

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can
    # concurrently download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model_kwargs = {
        "config": config,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    model = SparseAutoModel.text_generation_from_pretrained(
        model_name_or_path=model_args.model_name_or_path,
        model_type="model",
        **model_kwargs,
    )

    tokenizer_src = (
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else get_shared_tokenizer_src(model, None)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_src,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    make_eval_dataset = training_args.do_eval or data_args.num_export_samples > 0
    tokenized_datasets, raw_datasets = _get_tokenized_and_preprocessed_raw_datasets(
        config=config,
        data_args=data_args,
        model=model,
        raw_datasets=raw_datasets,
        tokenizer=tokenizer,
        make_eval_dataset=make_eval_dataset,
        main_process_func=training_args.main_process_first,
        do_train=training_args.do_train,
        do_predict=training_args.do_predict,
    )

    train_dataset = tokenized_datasets.get("train")
    eval_dataset = tokenized_datasets.get("validation")
    predict_dataset = tokenized_datasets.get("test")

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            _LOGGER.info(f"Sample {index} of training set: {train_dataset[index]}.")

    # Get the metric function
    metric = "perplexity"

    # placeholder
    def compute_metrics(p: EvalPrediction):
        return {f"perplexity: {0.0}"}

    # Data collator will default to DataCollatorWithPadding when the tokenizer is
    # passed to Trainer, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        model_state_path=model_args.model_name_or_path,
        recipe=training_args.recipe,
        metadata_args=metadata_args,
        recipe_args=training_args.recipe_args,
        args=training_args,
        data_args=data_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if make_eval_dataset else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        if not trainer.one_shot:
            metrics = train_result.metrics
            max_train_samples = (
                data_args.max_train_samples
                if data_args.max_train_samples is not None
                else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)

        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.save_state()
        trainer.save_optimizer_and_scheduler(training_args.output_dir)

    # Evaluation
    if training_args.do_eval and not trainer.one_shot:
        _LOGGER.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict and not trainer.one_shot:
        _LOGGER.info("*** Predict ***")
        results = trainer.predict(predict_dataset)
        metrics = results.metrics

        max_predict_samples = (
            data_args.max_predict_samples
            if data_args.max_predict_samples is not None
            else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "text-generation",
    }
    # Exporting Samples

    if data_args.num_export_samples > 0:
        trainer.save_sample_inputs_outputs(
            num_samples_to_export=data_args.num_export_samples
        )


def _get_tokenized_and_preprocessed_raw_datasets(
    config,
    data_args: DataTrainingArguments,
    model: Optional[Module],
    raw_datasets,
    tokenizer: transformers.PreTrainedTokenizerBase,
    make_eval_dataset: bool = False,
    do_predict: bool = False,
    do_train: bool = False,
    main_process_func=None,
):
    train_dataset = predict_dataset = eval_dataset = None
    config = model.config if model else config
    if not main_process_func:
        main_process_func = lambda desc: nullcontext(desc)  # noqa: E731

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence
        # length in each batch
        padding = False

    max_seq_length = data_args.max_seq_length
    if max_seq_length > tokenizer.model_max_length:
        _LOGGER.warning(
            f"The max_seq_length passed ({max_seq_length}) is larger than "
            f"the maximum length for the model ({tokenizer.model_max_length}). "
            f"Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        result = tokenizer(
            examples["text"], padding=False, max_length=max_seq_length, truncation=True
        )
        return result

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + max_seq_length]
                for i in range(0, total_length, max_seq_length)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with main_process_func(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=["text"],
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        raw_datasets = raw_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Grouping text",
        )

    train_dataset, eval_dataset, predict_dataset = _make_dataset_splits(
        raw_datasets, data_args, do_train, make_eval_dataset, do_predict
    )

    tokenized_datasets = {
        "train": train_dataset,
        "validation": eval_dataset,
        "test": predict_dataset,
    }
    return tokenized_datasets, raw_datasets


def _get_raw_dataset(
    data_args, cache_dir: Optional[str] = None, do_predict: bool = False
):
    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=cache_dir,
    )

    return raw_datasets


def _split_train_val(train_dataset, val_ratio):
    # Fixed random seed to make split consistent across runs with the same ratio
    seed = 42
    try:
        ds = train_dataset.train_test_split(
            test_size=val_ratio, stratify_by_column="label", seed=seed
        )
        train_ds = ds.pop("train")
        val_ds = ds.pop("test")
    except TypeError:
        X = list(range(len(train_dataset)))
        y = train_dataset["label"]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        for train_indices, test_indices in sss.split(X, y):
            train_ds = train_dataset.select(train_indices)
            val_ds = train_dataset.select(test_indices)

    return train_ds, val_ds


def _make_dataset_splits(
    raw_datasets, data_args, do_train, make_eval_dataset, do_predict
):
    """
    Make and return train, eval and predict splits. The eval split could
    come from either an explicit validation set, part of the training set,
    or the test set.
    """
    train_split = eval_split = predict_split = None
    if do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_split = raw_datasets["train"]
        if (
            data_args.max_train_samples is not None
            and len(train_split) > data_args.max_train_samples
        ):
            train_split = train_split.select(range(data_args.max_train_samples))

    if make_eval_dataset:
        if (
            "validation" not in raw_datasets
            and "validation_matched" not in raw_datasets
            and data_args.validation_ratio is None
            and data_args.eval_on_test is False
        ):
            raise ValueError(
                "--do_eval requires an explicit validation dataset, "
                "specified validation ratio, or eval_on_test"
            )
        elif "validation" in raw_datasets:
            if data_args.validation_ratio is not None:
                raise ValueError(
                    "validation_ratio cannot be specified when validation set exists"
                )
            if data_args.eval_on_test is True:
                if "test" not in raw_datasets:
                    raise ValueError("test split not found but eval_on_test is on")
                _LOGGER.info("eval_on_test: Evaluation on the test set")
                eval_split = raw_datasets["test"]
            else:
                eval_split = raw_datasets["validation"]
        elif data_args.validation_ratio is not None:
            if data_args.eval_on_test is True:
                raise ValueError(
                    "eval_on_test cannot be specified when validation_ratio is set"
                )
            train_split = raw_datasets["train"] if train_split is None else train_split
            train_split, eval_split = _split_train_val(
                train_split, data_args.validation_ratio
            )
        elif data_args.eval_on_test:
            if "test" not in raw_datasets:
                raise ValueError("test split not found but eval_on_test is on")
            eval_split = raw_datasets["test"]

        if (
            data_args.max_eval_samples is not None
            and len(eval_split) > data_args.max_eval_samples
        ):
            eval_split = eval_split.select(range(data_args.max_eval_samples))

    if do_predict:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_split = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_split = predict_split.select(range(data_args.max_predict_samples))

    return train_split, eval_split, predict_split


def get_tokenized_text_classification_dataset(
    data_args: DataTrainingArguments,
    tokenizer: transformers.PreTrainedTokenizerBase,
    model: Module,
    config,
    cache_dir: Optional[str] = None,
):
    """
    Utility method to get tokenized text classification dataset given at-least
    the tokenizer, model, and data_arguments

    :param data_args: Arguments pertaining to what data we are going to input
        our model for training and eval
    :param tokenizer: The tokenizer to use for tokenizing raw dataset
    :param config: The pretrained config used to load this model
    :param cache_dir: Local path to store the pretrained data from huggingface.co
    :returns: A dictionary containing tokenized_datasets
    """

    raw_datasets = _get_raw_dataset(data_args, cache_dir=cache_dir, do_predict=True)

    tokenized_datasets, _ = _get_tokenized_and_preprocessed_raw_datasets(
        config=config,
        data_args=data_args,
        model=model,
        raw_datasets=raw_datasets,
        tokenizer=tokenizer,
        make_eval_dataset=True,
    )

    return tokenized_datasets


if __name__ == "__main__":
    main()
