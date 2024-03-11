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
from pathlib import PosixPath

import datasets
import transformers
from transformers import AutoConfig, DefaultDataCollator, HfArgumentParser, set_seed

import sparseml.core.session as session_manager
from sparseml.core.framework import Framework
from sparseml.core.recipe import Recipe, StageRunType
from sparseml.pytorch.model_load.helpers import (
    apply_recipe_structure_to_model,
    fallback_to_cpu,
    get_session_model,
    parse_dtype,
)
from sparseml.transformers import SparseAutoTokenizer
from sparseml.transformers.finetune.data.data_args import DataTrainingArguments
from sparseml.transformers.finetune.model_args import ModelArguments
from sparseml.transformers.finetune.runner import StageRunner
from sparseml.transformers.finetune.trainer import Trainer
from sparseml.transformers.finetune.training_args import TrainingArguments
from sparseml.transformers.utils import SparseAutoModel, get_shared_tokenizer_src
from sparseml.transformers.utils.helpers import detect_last_checkpoint


_LOGGER: logging.Logger = logging.getLogger(__name__)

metadata_args = [
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "max_seq_length",
    "fp16",
]


def train(**kwargs):
    """
    CLI entrypoint for running training
    """
    model_args, data_args, training_args = parse_args(**kwargs)
    training_args.do_train = True
    main(model_args, data_args, training_args)


def eval(**kwargs):
    """
    CLI entrypoint for running evaluation
    """
    model_args, data_args, training_args = parse_args(**kwargs)
    training_args.do_eval = True
    main(model_args, data_args, training_args)


def oneshot(**kwargs):
    """
    CLI entrypoint for running oneshot calibration
    """
    model_args, data_args, training_args = parse_args(**kwargs)
    training_args.do_oneshot = True
    main(model_args, data_args, training_args)


# alias
one_shot = oneshot


def apply(**kwargs):
    """
    CLI entrypoint for any of training, eval, predict or oneshot
    """
    model_args, data_args, training_args = parse_args(**kwargs)
    training_args.run_stages = True
    main(model_args, data_args, training_args)


def compress(**kwargs):
    apply(**kwargs)


def load_dataset(dataset_name: str, **kwargs):

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_dict(kwargs)
    data_args["dataset_name"] = dataset_name


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
        if not isinstance(training_args.recipe_args, dict):
            arg_dict = {}
            for recipe_arg in training_args.recipe_args:
                key, value = recipe_arg.split("=")
                arg_dict[key] = value
            training_args.recipe_args = arg_dict

    # when set to true in FSDP mode this causes issues, the model arguments show up
    # as *args and **kwargs so all columns get removed
    training_args.remove_unused_columns = False

    return model_args, data_args, training_args


def intialize_model_from_path(
    model_args: ModelArguments,
    training_args: TrainingArguments,
):

    last_checkpoint = detect_last_checkpoint(training_args, model_args=model_args)
    # Load pretrained model
    # The .from_pretrained methods guarantee that only one local process can
    # concurrently download model & vocab.
    model_path = model_args.model
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    teacher_config = (
        AutoConfig.from_pretrained(
            model_args.distill_teacher,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        if model_args.distill_teacher
        else None
    )

    model_path = (
        last_checkpoint or model_args.model
        if hasattr(model_args, "model")
        else model_args.model_name_or_path
    )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Fallback to CPU if GPU requested and not available
    training_args.oneshot_device = fallback_to_cpu(training_args.oneshot_device)

    # Trainer handles device assignment for FSDP and training, don't do mapping here
    # if running oneshot outside of FSDP, apply user device settings
    device_map = None
    fsdp_enabled = os.environ.get("ACCELERATE_USE_FSDP", "false") == "true"
    if not fsdp_enabled and training_args.do_oneshot:
        device_map = training_args.oneshot_device
        _LOGGER.warning(f"Moving {model_path} to device {device_map} for One-Shot")
    elif not fsdp_enabled:
        device_map = "auto"
    model_kwargs = {
        "config": config,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "torch_dtype": parse_dtype(model_args.precision),
        "device_map": device_map,
    }
    teacher_device_map = None if fsdp_enabled else "auto"
    teacher_kwargs = {
        "config": teacher_config,
        "cache_dir": model_args.cache_dir,
        "use_auth_token": True if model_args.use_auth_token else None,
        "torch_dtype": parse_dtype(model_args.precision),
        "device_map": teacher_device_map,
    }
    # this calls from_pretrained under the hood so should be FSDP safe
    model = SparseAutoModel.text_generation_from_pretrained(
        model_name_or_path=model_path,
        sequence_length=None,  # use model default
        **model_kwargs,
    )

    teacher = (
        SparseAutoModel.text_generation_from_pretrained(
            model_name_or_path=model_args.distill_teacher,
            sequence_length=None,  # use model default
            **teacher_kwargs,
        )
        if model_args.distill_teacher is not None
        else None
    )

    return teacher, model_path, model


def initialize_tokenizer_from_path(model_args, model, teacher):
    tokenizer_src = model_args.tokenizer
    tokenizer_src = tokenizer_src or get_shared_tokenizer_src(model, teacher)
    tokenizer = SparseAutoTokenizer.from_pretrained(
        tokenizer_src,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    return tokenizer


def main(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
):
    """
    Main entrypoint for finetuning text generation models. A model can be loaded from
    Hugging Face or disk, and resuming training from a checkpoint is supported.

    Lifecycle:
        - SparseAutoModel.text_generation_from_pretrained if model provided as
            string for model and teacher
        - SparseAutoTokenizer.from_pretrained() if tokenizer provided as
            string for tokenizer
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

    # Setup based on stage types if running stage mode
    if training_args.run_stages and training_args.recipe is not None:
        recipe_obj = Recipe.create_instance(training_args.recipe)
        for stage in recipe_obj.stages:
            run_type = stage.infer_run_type()
            if run_type is StageRunType.ONESHOT:
                training_args.do_oneshot = True
            elif run_type is StageRunType.TRAIN:
                training_args.do_train = True

    # Summary on each process
    _LOGGER.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16}"
    )
    _LOGGER.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    teacher = model_args.distill_teacher
    model_path = None
    model = model_args.model
    # Load tokenizer
    # distill TODO: support for different tokenizer for teacher?
    tokenizer = model_args.tokenizer

    if isinstance(model, str) or isinstance(model, PosixPath):
        (teacher, model_path, model) = intialize_model_from_path(
            model_args,
            training_args,
        )

    if teacher is not None:
        teacher.eval()

    if isinstance(tokenizer, str) or tokenizer is None:
        tokenizer = initialize_tokenizer_from_path(model_args, model, teacher)

    # setup new SparseSession unless user requests otherwise
    if training_args.clear_sparse_session:
        session_manager.create_session()
        session_manager.active_session().reset()
    session_manager.pre_initialize_structure(model=model, framework=Framework.pytorch)

    # intialize session manager
    apply_recipe_structure_to_model(model, None, model_path)

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

    # Initialize our Trainer
    data_collator = DefaultDataCollator()
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
    apply()
