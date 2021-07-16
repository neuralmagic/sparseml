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

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import numpy

import datasets
from datasets import load_dataset
import wandb

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from sparseml_utils import SparseMLTrainer, export_model

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
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
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
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
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
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
    block_size: Optional[int] = field(
        default=1024,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

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

    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([numpy.prod(p.size()) for p in model_parameters])
    logger.info("Model has %s parameters", params)
    model.resize_token_embeddings(len(tokenizer))

    teacher_model = None
    if model_args.distill_teacher is not None:
        teacher_model = AutoModelForCausalLM.from_pretrained(
            model_args.distill_teacher,
            from_tf=bool(".ckpt" in model_args.distill_teacher),
            config=config,
            cache_dir=model_args.cache_dir,
        )
        teacher_model_parameters = filter(lambda p: p.requires_grad, teacher_model.parameters())
        params = sum([numpy.prod(p.size()) for p in teacher_model_parameters])
        logger.info("Teacher Model has %s parameters", params)
        teacher_model.resize_token_embeddings(len(tokenizer))
    
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

    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
            )
        return output
        
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    if training_args.do_train or training_args.do_eval:
        if data_args.dataset_name is not None:
            raw_datasets = load_dataset(
                data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
            )
        else:
            data_files = {}
            if data_args.train_file is not None:
                data_files["train"] = data_args.train_file
            if data_args.validation_file is not None:
                data_files["validation"] = data_args.validation_file
            extension = (
                data_args.train_file.split(".")[-1]
                if data_args.train_file is not None
                else data_args.validation_file.split(".")[-1]
            )
            if extension == "txt":
                extension = "text"
            raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
        
        if training_args.do_train:
            column_names = raw_datasets["train"].column_names
        else:
            column_names = raw_datasets["validation"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]
        import pdb
        #pdb.set_trace()
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    trainer =  SparseMLTrainer(
        model=model,
        recipe=data_args.recipe,
        teacher=teacher_model,
        distill_hardness=model_args.distill_hardness,
        distill_temperature=model_args.distill_temperature,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    wandb.init(project='gpt-distill', entity='spacemanidol')
    if training_args.do_train:
        logger.info("*** Train ***")
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

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub:
        logger.info("*** Push to hub ***")
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
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


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()