# neuralmagic: no copyright
# flake8: noqa
# fmt: off
# isort: skip_file
#!/usr/bin/env python
# coding=utf-8
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

"""
Example script for integrating spaseml with the transformers library to perform model distillation and pruning on GLUE tasks. 
This script is addopted from hugging face's implementation for the GLUEDataset. 
Hugging Face's original implementation is regularly updated and can be found at https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue.py
This script will:
- Load transformer based models
- Load a sparseml training and pruning optimizer
- Train on Target GLUE Task
- Evaluate on GLUE
- Export model to onnx.
##########
Command help:
usage: run_glue.py [-h] \
    [--teacher_model_name_or_path] \
    [--student_model_name_or_path] \
    [--task_name] \
    [--temperature] \
    [--distill_hardness] \
    [--dataset_name]  \
    [--num_train_epochs] \
    [--do_train] \
    [--do_eval] \
    [--per_device_train_batch_size] \
    [--per_device_eval_batch_size] \
    [--learning_rate]\
    [--output_dir] \
    [--overwrite_output_dir] \
    [--cache_dir]\
    [--preprocessing_num_workers] \
    [--seed] \
    [--nm_prune_config] \
    [--do_onnx_export] \
    [--onnx_export_path] \
    [--layers_to_keep] \

Train, prune, and evaluate a transformer base question answering model on squad. 
    -h, --help            show this help message and exit
    --teacher_model_name_or_path    The name or path of model which will be used for distilation.
                                    Note, this model needs to be trained for QA task already.
    --student_model_name_or_path    The path to the transformers model you wish to train
                                    or the name of the pretrained language model you wish
                                    to use. ex: bert-base-uncased.
    --task_name                     The name of the GLUE task which the model with train and evalute on. 
    --temperature                   Hyperparameter which controls model distilation 
    --distill_hardness              Hyperparameter which controls how much of the loss comes from teacher vs training labels
    --dataset_name                  The name of which dataset you want to use to train or
                                    your model. ex: squad for using SQuAD.
    --num_train_epochs              Paramater to control how many training epochs you wish
                                    your model to train.
    --do_train                      Boolean denoting if the model should be trained
                                    or not. Default is false.
    --do_eval                       Boolean denoting if the model should be evaluated
                                    or not. Default is false.
    --per_device_train_batch_size   Size of each training batch based on samples per GPU. 
                                    24 will fit in a 11gb GPU, 32 in a 16gb.
    --per_device_eval_batch_size    Size of each training batch based on samples per GPU. 
                                    24 will fit in a 11gb GPU, 32 in a 16gb.
    --learning_rate                 Learning rate initial float value. ex: 3e-5.
    --output_dir                    Path which model checkpoints and paths should be saved.
    --overwrite_output_dir          Boolean to define if the 
    --cache_dir                     Directiory which cached transformer files(datasets, models
                                    , tokenizers) are saved for fast loading. 
    --preprocessing_num_workers     The amount of cpu workers which are used to process datasets
    --seed                          Int which determines what random seed is for training/shuffling
    --nm_prune_config               Path to the neural magic prune configuration file. examples can
                                    be found in prune_config_files but are customized for bert-base-uncased. 
    --do_onnx_export                Boolean denoting if the model should be exported to onnx
    --onnx_export_path              Path where onnx model path will be exported. ex: onnx-export
    --layers_to_keep                Number of layers to keep from original model. Layers are dropped before training
    --max_seq_length                Int for the max sequence length to be parsed for glue tasks ex: 128 tokens.

##########
Example command for training a 95% sparse BERT SQUAD model for 1 epoch without distilation on the Quora Duplicate Question Task:
python examples/transformers/run_glue.py \
    --teacher_model_name_or_path NONE
    --student_model_name_or_path bert-base-uncased \
    --task_name QQP
    --dataset_name squad \
    --num_train_epochs 1 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 12 \
    --learning_rate 3e-5 \
    --max_seq_length 128 \
    --doc_stride 128 \
    --output_dir 95sparsity1epoch/ \
    --overwrite_output_dir \
    --cache_dir cache \
    --preprocessing_num_workers 8 \
    --seed 42 \
    --nm_prune_config prune_config_files/95sparsity1epoch.yaml \
    --do_onnx_export \
    --onnx_export_path 95sparsity1epoch/ \
    --distill_hardness 1.0 \
    --temperature 2.0 \
    --layers_to_keep 12 \
"""

import argparse
import logging

import numpy as np
import torch

from transformers import (
    set_seed,
    HfArgumentParser,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    default_data_collator,
    TrainingArguments,
)

import logging
import os
import json
import sys
from dataclasses import dataclass, field
from typing import Optional
import random
import math
import numpy

import pdb
import wandb
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_metric

import transformers
from filelock import FileLock

from transformers.testing_utils import CaptureLogger
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version

from sparseml_utils import SparseMLTrainer, export_model

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    distill_teacher: Optional[str] = field(
        default=None, metadata={"help": "Path to a pretrained Doc2Query Model to be used for distillation"}
    )
    distill_temperature: Optional[float] = field(
        default=2.0, metadata={"help": "Temperature applied to teacher softmax for distillation."}
    )
    distill_hardness: Optional[float] = field(
        default=0.5, metadata={"help": "Proportion of loss coming from teacher model."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    recipe: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a SparseML sparsification recipe, see https://github.com/neuralmagic/sparseml "
                  "for more information"},
    )
    onnx_export_path: Optional[str] = field(
        default=None, metadata={"help": "The filename and path which will be where onnx model is outputed"}
    )
    repetition_penalty: Optional[float] = field(
        default=1.0, metadata={"help": "penalty for repetition which can be useful in generative models"}
    )
    prompt: Optional[str] = field(
        default=None, metadata={"help": "Generation prompt."}
    )
    prefix: Optional[str] = field(
        default=None, metadata={"help": "Text added prior to input."}
    )
    num_return_sequences: Optional[int] = field(
        default=1, metadata={"help": "How many sequences to return"}
    )
    dataset_name: Optional[str] = field(
        default='wikitext', metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default='wikitext-103-raw-v1', metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the model on "
            "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    max_source_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)

    if data_args.dataset_name is not None: #Need to upload datasets
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)
    else:
        data_files = {}
        if data_args.train_file is not None and training_args.do_train:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None and training_args.do_eval:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([numpy.prod(p.size()) for p in model_parameters])
    logger.info("Model has %s parameters", params)
    model.resize_token_embeddings(len(tokenizer))

    teacher_model = None
    if model_args.distill_teacher is not None:
        teacher_model = AutoModel.from_pretrained(
            model_args.distill_teacher,
            from_tf=bool(".ckpt" in model_args.distill_teacher),
            config=config,
            cache_dir=model_args.cache_dir,
        )
        teacher_model_parameters = filter(lambda p: p.requires_grad, teacher_model.parameters())
        params = sum([numpy.prod(p.size()) for p in teacher_model_parameters])
        logger.info("Teacher Model has %s parameters", params)
        teacher_model.resize_token_embeddings(len(tokenizer))

    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return
    text_column_name = "text" if "text" in column_names else column_names[0]
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        return output

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)    

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )
        train_dataset = train_dataset.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,  
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Grouping texts in chunks of {block_size} for train",
        )

    if training_args.do_eval:
        if "validation" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        validation_dataset = datasets["validation"]
        validation_dataset = validation_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on Validation dataset",
        )
        validation_dataset = validation_dataset.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Grouping texts in chunks of {block_size} for validation",
        )

    trainer = SparseMLTrainer(
        model=model,
        recipe=data_args.recipe,
        teacher=teacher_model,
        distill_hardness=model_args.distill_hardness,
        distill_temperature=model_args.distill_temperature,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=validation_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    wandb.init(project='gpt-distill', entity='spacemanidol')
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(validation_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(validation_dataset))
        print(metrics)
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tags": "summarization"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)
    
    if data_args.onnx_export_path:
        logger.info("*** Export to ONNX ***")
        eval_dataloader = trainer.get_eval_dataloader(eval_dataset)
        export_model(model, eval_dataloader, data_args.onnx_export_path)
    return results

def _mp_fn(index):
    main()

if __name__ == "__main__":
    main()