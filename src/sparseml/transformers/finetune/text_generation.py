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


import logging
import os
import random
from typing import Tuple

import datasets
import transformers
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    default_data_collator,
    set_seed,
)

from sparseml.transformers.finetune import Trainer, TrainingArguments
from sparseml.transformers.finetune.data import TextGenerationDataset
from sparseml.transformers.finetune.data.data_args import DataTrainingArguments
from sparseml.transformers.finetune.data.data_helpers import make_dataset_splits
from sparseml.transformers.finetune.helpers import apply_recipe_structure_to_model
from sparseml.transformers.finetune.model_args import ModelArguments
from sparseml.transformers.utils import SparseAutoModel, get_shared_tokenizer_src
from sparseml.transformers.utils.helpers import detect_last_checkpoint


_LOGGER: logging.Logger = logging.getLogger(__name__)

metadata_args = [
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "fp16",
]


def run_train(**kwargs):
    """
    CLI entrypoint for running training
    """
    model_args, data_args, training_args = parse_args(**kwargs)
    training_args.do_train = True
    main(model_args, data_args, training_args)


def run_eval(**kwargs):
    """
    CLI entrypoint for running evaluation
    """
    model_args, data_args, training_args = parse_args(**kwargs)
    training_args.do_eval = True
    main(model_args, data_args, training_args)


def run_general(**kwargs):
    """
    CLI entrypoint for any of training, eval or predict
    """
    model_args, data_args, training_args = parse_args(**kwargs)
    main(model_args, data_args, training_args)


def parse_args(**kwargs):
    """
    Parses kwargs by grouping into model, data or training arg groups:
        * model_args in src/sparseml/transformers/finetune/model_args.py
        * data_args in src/sparseml/transformers/finetune/data/data_args.py
        * training_args in src/sparseml/transformers/finetune/training_args.py
    """
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if not kwargs:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    else:
        model_args, data_args, training_args = parser.parse_dict(kwargs)

    return model_args, data_args, training_args


def main(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
):
    """
    Main entrypoint for finetuning text generation models. A model can be loaded from
    Hugging Face or disk, and resuming training from a checkpoint is supported.

    Lifecycle:
        - get_last_checkpoint() [Optional]
        - AutoModel.text_generation_from_pretrained()
        - AutoTokenizer.from_pretrained()
        - TextGenerationDataset.load_from_registry()
        - Trainer()
            - SessionMixIn()
            - HFTransformersTrainer()
        - train() and/or evaluate() and/or predict()

    :param model_args: Arguments pertaining to which model/config/tokenizer we are
    going to fine-tune from
    :param data_args: Arguments pertaining to what data we are going to input our model
    for training and eval
    :param training_args: Arguments pertaining to training loop configuration
    """

    # Setup logging
    log_level = training_args.get_process_log_level()
    _LOGGER.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Summary on each process
    _LOGGER.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16}"
    )
    _LOGGER.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = detect_last_checkpoint(training_args)
    model_path = last_checkpoint or model_args.model_name_or_path

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model
    # The .from_pretrained methods guarantee that only one local process can
    # concurrently download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_path,
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
    teacher_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    # this calls from_pretrained under the hood so should be FSDP safe
    model, teacher = SparseAutoModel.text_generation_from_pretrained_distil(
        model_name_or_path=model_path,
        teacher_name_or_path=training_args.distill_teacher,
        model_kwargs=model_kwargs,
        teacher_kwargs=teacher_kwargs,
    )

    # initialize structure of input model from recipe if needed
    recipe_path = os.path.join(model_path, "recipe.yaml")
    if last_checkpoint is not None and training_args.recipe is None:
        training_args.recipe = recipe_path  # continue from checkpoint recipe
    else:
        if not os.path.exists(recipe_path):
            _LOGGER.warning(f"No recipes were applied for {model_path}.")
        else:
            _LOGGER.warning(f"Applying recipe {recipe_path} to {model_path}")
            apply_recipe_structure_to_model(model, recipe_path, model_path)

    # Load tokenizer
    # distill TODO: support for different tokenizer for teacher?
    tokenizer_src = (
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else get_shared_tokenizer_src(model, teacher)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_src,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Load datasets
    # TODO: will any of this cause problems with FSDP?
    train_dataset, eval_dataset, predict_dataset = _load_split_datasets(
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        tokenizer=tokenizer,
    )

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            _LOGGER.info(f"Sample {index} of training set: {train_dataset[index]}.")

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
        teacher=teacher,
        model_state_path=model_path,
        recipe=training_args.recipe,
        metadata_args=metadata_args,
        recipe_args=training_args.recipe_args,
        args=training_args,
        data_args=data_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train(checkpoint, training_args.output_dir, train_dataset, trainer)

    # Evaluation
    if training_args.do_eval:
        evaluate(eval_dataset, trainer)

    # Prediction
    if training_args.do_predict:
        predict(predict_dataset, trainer)


def train(checkpoint: str, output_dir: str, train_dataset: Dataset, trainer: Trainer):
    """
    Run trainer's training loop on train_dataset, saving the resulting model to
    output_dir

    :param checkpoint: Optional checkpoint to resume from
    :param output_dir: Where to output trained model and recipe
    :param train_dataset: Dataset to run training on
    :param trainer: Trainer object to run training with
    """
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # this includes saving the state, optimizer and scheduler
    trainer.save_model()


def evaluate(eval_dataset: Dataset, trainer: Trainer):
    """
    Run trainer's evaluation loop on eval_dataset, logging the desired metrics

    :param eval_dataset: Dataset to run evaluation on
    :param trainer: Trainer object to run evaluation with
    """
    _LOGGER.info("*** Evaluate ***")
    metrics = trainer.evaluate(eval_dataset)

    metrics["eval_samples"] = len(eval_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


def predict(predict_dataset: Dataset, trainer: Trainer):
    """
    Run trainer's prediction loop on predict_dataset, logging the desired metrics

    :param eval_dataset: Dataset to run prediction on
    :param trainer: Trainer object to run prediction with
    """
    _LOGGER.info("*** Predict ***")
    results = trainer.predict(predict_dataset)
    metrics = results.metrics

    metrics["predict_samples"] = len(predict_dataset)
    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)


def _load_split_datasets(
    data_args: DataTrainingArguments,
    model_args: ModelArguments,
    training_args: TrainingArguments,
    tokenizer: AutoTokenizer,
) -> Tuple[Dataset, Dataset, Dataset]:
    splits = data_args.splits
    tokenized_datasets = {}
    if data_args.splits is None:
        splits = {"all": None}
    for split_name, split_str in splits.items():
        dataset_manager = TextGenerationDataset.load_from_registry(
            data_args.dataset_name,
            data_args=data_args,
            split=split_str,
            tokenizer=tokenizer,
        )
        raw_dataset = dataset_manager.get_raw_dataset(model_args.cache_dir)
        tokenized_dataset = dataset_manager.tokenize_and_process(raw_dataset)
        tokenized_datasets[split_name] = tokenized_dataset

    tokenized_datasets = make_dataset_splits(
        tokenized_datasets,
        training_args.do_train,
        training_args.do_eval,
        training_args.do_predict,
    )

    train_dataset = tokenized_datasets.get("train")
    eval_dataset = tokenized_datasets.get("validation")
    predict_dataset = tokenized_datasets.get("test")

    return train_dataset, eval_dataset, predict_dataset


if __name__ == "__main__":
    run_general()
