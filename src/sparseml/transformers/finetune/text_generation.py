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

import datasets
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    default_data_collator,
    set_seed,
)

from sparseml.pytorch.model_load.helpers import (
    apply_recipe_structure_to_model,
    fallback_to_cpu,
    get_session_model,
    parse_dtype,
)
from sparseml.transformers.finetune import Trainer, TrainingArguments
from sparseml.transformers.finetune.data.data_args import DataTrainingArguments
from sparseml.transformers.finetune.model_args import ModelArguments
from sparseml.transformers.finetune.runner import StageRunner
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


def run_oneshot(**kwargs):
    """
    CLI entrypoint for running oneshot calibration
    """
    model_args, data_args, training_args = parse_args(**kwargs)
    training_args.do_oneshot = True
    main(model_args, data_args, training_args)


def run_general(**kwargs):
    """
    CLI entrypoint for any of training, eval, predict or oneshot
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

    if training_args.recipe_args is not None:
        arg_dict = {}
        for recipe_arg in training_args.recipe_args:
            key, value = recipe_arg.split("=")
            arg_dict[key] = value
        training_args.recipe_args = arg_dict

    if training_args.run_stages:
        training_args.do_oneshot = True
        training_args.do_train = True

    # when set to true in FSDP mode this causes issues, the model arguments show up
    # as *args and **kwargs so all columns get removed
    training_args.remove_unused_columns = False

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
        - AutoTokenizer.from_pretrained_distil()
        - StageRunner.populate_datasets()
        - Trainer()
            - SessionMixIn()
            - HFTransformersTrainer()
        - StageRunner.train() and/or evaluate() and/or predict() and/or oneshot()

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
    last_checkpoint = detect_last_checkpoint(training_args, model_args=model_args)
    model_path = last_checkpoint or model_args.model_name_or_path

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Fallback to CPU if GPU requested and not available
    training_args.oneshot_device = fallback_to_cpu(training_args.oneshot_device)

    # Load pretrained model
    # The .from_pretrained methods guarantee that only one local process can
    # concurrently download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    teacher_config = (
        AutoConfig.from_pretrained(
            training_args.distill_teacher,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        if training_args.distill_teacher
        else None
    )

    model_kwargs = {
        "config": config,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "torch_dtype": parse_dtype(model_args.precision),
        "device_map": training_args.oneshot_device,
    }
    teacher_kwargs = {
        "config": teacher_config,
        "cache_dir": model_args.cache_dir,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    # this calls from_pretrained under the hood so should be FSDP safe
    model = SparseAutoModel.text_generation_from_pretrained(
        model_name_or_path=model_path,
        sequence_length=None,  # use model default
        **model_kwargs,
    )

    teacher = (
        SparseAutoModel.text_generation_from_pretrained(
            model_name_or_path=training_args.distill_teacher,
            sequence_length=None,  # use model default
            **teacher_kwargs,
        )
        if training_args.distill_teacher is not None
        else None
    )

    # initialize structure of input model from recipe if needed
    recipe_path = os.path.join(model_path, "recipe.yaml")
    if last_checkpoint is not None and training_args.recipe is None:
        training_args.recipe = recipe_path  # continue from checkpoint recipe
        apply_recipe_structure_to_model(model, None, model_path)
    else:
        if not os.path.exists(recipe_path):
            _LOGGER.warning(f"No recipes were applied for {model_path}.")
            apply_recipe_structure_to_model(model, None, model_path)
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
    stage_runner = StageRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        model=model,
    )
    stage_runner.populate_datasets(tokenizer=tokenizer)
    train_dataset = stage_runner.get_dataset_split("train")
    eval_dataset = stage_runner.get_dataset_split("validation")
    calib_dataset = stage_runner.get_dataset_split("calibration")

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
        model_init=get_session_model,
        teacher=teacher,
        model_state_path=model_path,
        recipe=training_args.recipe,
        metadata_args=metadata_args,
        recipe_args=training_args.recipe_args,
        args=training_args,
        data_args=data_args,
        train_dataset=train_dataset or calib_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    if trainer.is_fsdp_enabled:
        trainer._prepare_model_for_fsdp()
    stage_runner.trainer = trainer

    # alternating Training/One-shot
    if training_args.run_stages:
        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        stage_runner.run_sequential_stages(checkpoint)

        # exit immediately
        return

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        stage_runner.train(checkpoint)

    # One Shot
    if training_args.do_oneshot:
        stage_runner.one_shot()

    # Evaluation
    if training_args.do_eval:
        stage_runner.evaluate()

    # Prediction
    if training_args.do_predict:
        stage_runner.predict()


if __name__ == "__main__":
    run_general()
