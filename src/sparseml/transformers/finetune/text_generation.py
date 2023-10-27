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
from transformers.trainer_utils import get_last_checkpoint

from sparseml.transformers.finetune import Trainer, TrainingArguments
from sparseml.transformers.finetune.data import TextGenerationDataset
from sparseml.transformers.finetune.data.data_args import DataTrainingArguments
from sparseml.transformers.finetune.helpers import apply_recipe_structure_to_model
from sparseml.transformers.finetune.model_args import ModelArguments
from sparseml.transformers.utils import SparseAutoModel, get_shared_tokenizer_src


_LOGGER: logging.Logger = logging.getLogger(__name__)

metadata_args = [
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "fp16",
]


def main(**kwargs):
    # model_args in src/sparseml/transformers/finetune/model_args.py
    # data_args in src/sparseml/transformers/finetune/data/data_args.py
    # training_args in src/sparseml/transformers/finetune/training_args.py
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

    # Summary on each process
    _LOGGER.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16}"
    )
    _LOGGER.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    # TODO: test this checkpoint loading after model saving is working
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

    # Load pretrained model
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
    # this calls from_pretrained under the hood so should be FSDP safe
    model = SparseAutoModel.text_generation_from_pretrained(
        model_name_or_path=model_args.model_name_or_path,
        model_type="model",
        **model_kwargs,
    )

    # initialize structure of input model from recipe if needed
    model_path = model_args.model_name_or_path
    recipe_path = os.path.join(model_path, "recipe.yaml")
    if not os.path.exists(recipe_path):
        _LOGGER.warning(f"No recipes were applied for {model_path}.")
    else:
        _LOGGER.warning(f"Applying recipe {recipe_path} to {model_path}")
        apply_recipe_structure_to_model(model, recipe_path, model_path)

    # Load tokenizer
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

    # Load datasets
    # TODO: will any of this cause problems with FSDP?
    do_eval = training_args.do_eval or data_args.num_export_samples > 0
    dataset_manager = TextGenerationDataset.load_from_registry(
        data_args.dataset_name, data_args=data_args, tokenizer=tokenizer
    )
    raw_dataset = dataset_manager.get_raw_dataset(model_args.cache_dir)
    tokenized_datasets = dataset_manager.tokenize_and_process(raw_dataset)
    tokenized_datasets = dataset_manager.make_dataset_splits(
        tokenized_datasets, training_args.do_train, do_eval, training_args.do_predict
    )
    train_dataset = tokenized_datasets.get("train")
    eval_dataset = tokenized_datasets.get("validation")
    predict_dataset = tokenized_datasets.get("test")

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
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        trainer.save_model()
        trainer.save_state()
        trainer.save_optimizer_and_scheduler(training_args.output_dir)

    # Evaluation
    if training_args.do_eval:
        _LOGGER.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        _LOGGER.info("*** Predict ***")
        results = trainer.predict(predict_dataset)
        metrics = results.metrics

        metrics["predict_samples"] = len(predict_dataset)
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


if __name__ == "__main__":
    main()