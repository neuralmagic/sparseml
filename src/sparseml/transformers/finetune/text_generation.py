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
from typing import Optional, Dict, Any
import torch

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
import sparseml.core.session as session_manager
from sparseml.core.framework import Framework
from sparseml.pytorch.sparsification.quantization.helpers import (
    initialize_channel_wise_scale_zp,
)
from sparseml.transformers.finetune.data.data_args import DataTrainingArguments
from sparseml.transformers.finetune.data import TextGenerationDataset
from sparseml.transformers.finetune.model_args import ModelArguments

_LOGGER: logging.Logger = logging.getLogger(__name__)

metadata_args = [
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "fp16",
]


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

    model = model.train()
    model_path = model_args.model_name_or_path
    recipe_path = os.path.join(model_path, "recipe.yaml")
    if not os.path.exists(recipe_path):
        _LOGGER.warning(
            f"No recipes were applied for {model_path}."
        )
    else:
        orig_state_dict = model.state_dict()

        session_manager.create_session()
        session_manager.pre_initialize_structure(
            model=model, recipe=recipe_path, framework=Framework.pytorch
        )

        session = session_manager.active_session()
        num_stages = len(session.lifecycle.recipe_container.compiled_recipe.stages)
        msg = (
            "an unstaged recipe"
            if num_stages == 1
            else f"a staged recipe with {num_stages} stages"
        )
        _LOGGER.info(f"Applied {msg} to the model at {model_path}")

        # reload the state dict for the model now that architecture matches expected
        if _reload_model_state(model, model_path, orig_state_dict):
            _LOGGER.info(
                "Reloaded model state after SparseML recipe structure modifications "
                f"from {model_path}"
            )
        session_manager.active_session().reset()

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

    do_eval = training_args.do_eval or data_args.num_export_samples > 0

    dataset_manager = TextGenerationDataset.load_from_registry(
        data_args.dataset_name,
        data_args=data_args,
        tokenizer=tokenizer
    )
    raw_dataset = dataset_manager.get_raw_dataset(model_args.cache_dir)
    tokenized_datasets = dataset_manager.tokenize_and_process(raw_dataset)
    tokenized_datasets = dataset_manager.make_dataset_splits(tokenized_datasets, training_args.do_train, do_eval, training_args.do_predict)

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
        eval_dataset=eval_dataset if do_eval else None,
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

def _reload_model_state(model, load_path: str, orig_state_dict: Dict[str, Any]):
    """
    Reload the weights after model arch changes due to recipe application
    Return True if weights are successfully reloaded; False otherwise
    """
    invalid_load_path = not load_path or not os.path.isdir(load_path)
    files = os.listdir(load_path) if not invalid_load_path else []
    weight_files = [
        os.path.join(load_path, os.path.basename(f))
        for f in files
        if f.startswith("pytorch_model") and f.endswith("bin")
    ]
    if not weight_files:
        _LOGGER.warning(
            "Model state was not reloaded for SparseML: "
            f"could not find model weights for {load_path}"
        )
        return False

    # PerChannel quantization observers initialize variables
    # to dummy shapes that do not match the ones saved in
    # state_dict.
    # Need to reshape these variables in order to load state_dict
    # properly.
    initialize_channel_wise_scale_zp(model)

    current_state_dict = model.state_dict()

    if set(orig_state_dict.keys()) == set(current_state_dict):
        # no change in keys, ignore reload
        return False

    # change in keys due to architecture changes, reload statedict
    loaded_state_dict = {}
    for f in weight_files:
        dd = torch.load(f, map_location="cpu")
        loaded_state_dict.update(dd)

    _, missing, unexpected, mismatched, _, _ = model._load_pretrained_model(
        model=model,
        state_dict=loaded_state_dict,
        loaded_keys=list(loaded_state_dict.keys()),
        resolved_archive_file=None,
        pretrained_model_name_or_path=load_path,
        _fast_init=False,
    )

    if missing:
        _LOGGER.warning(
            "Missing keys found when reloading model state for SparseML recipe:"
            f"{missing}"
        )

    if unexpected:
        _LOGGER.warning(
            f"Unexpected keys found when reloading model state for SparseML recipe:"
            f"{unexpected}"
        )

    if mismatched:
        _LOGGER.warning(
            f"Mismatched keys found when reloading model state for SparseML recipe:"
            f"{mismatched}"
        )

    total_loaded = len(current_state_dict) - (len(missing) if len(missing) else 0)
    _LOGGER.info(
        f"Reloaded {total_loaded} model params for SparseML Recipe from {load_path}"
    )
    SparseAutoModel.log_model_load(
        model,
        load_path,
        model_type="student",
        delayed_load=False,
    )
    return True


if __name__ == "__main__":
    main()
