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
Example script for integrating spaseml with the transformers library to perform model distillation.
This script is addopted from hugging face's implementation for Question Answering on the SQUAD Dataset.
Hugging Face's original implementation is regularly updated and can be found at https://github.com/huggingface/transformers/blob/master/examples/question-answering/run_qa.py
This script will:
- Load transformer based models
- Load a sparseml training and pruning optimizer
- Train on SQUAD
- Evaluate on SQUAD
- Export model to onnx.
##########
Command help:
usage: run_distill_qa.py [-h] \
    [--teacher_model_name_or_path] \
    [--student_model_name_or_path] \
    [--temperature] \
    [--distill_hardness] \
    [--dataset_name]  \
    [--num_train_epochs] \
    [--do_train] \
    [--do_eval] \
    [--per_device_train_batch_size] \
    [--per_device_eval_batch_size] \
    [--learning_rate]\
    [--max_seq_length]\
    [--doc_stride]\
    [--output_dir] \
    [--overwrite_output_dir] \
    [--cache_dir]\
    [--preprocessing_num_workers] \
    [--seed] 42 \
    [--nm_prune_config] \
    [--do_onnx_export] \
    [--onnx_export_path] \
    [--layers_to_keep] \

Train, prune, and evaluate a transformer base question answering model on squad.
    -h, --help            show this help message and exit
    --teacher_model_name_or_path    The name or path of model which will be used for distilation.
                                    Note, this model needs to be trained for QA task already.
    --student_model_name_or_path    The name or path of the model wich will be trained using distilation.
    --temperature                   Hyperparameter which controls model distilation
    --distill_hardness              Hyperparameter which controls how much of the loss comes from teacher vs training labels
    --model_name_or_path            The path to the transformers model you wish to train
                                    or the name of the pretrained language model you wish
                                    to use. ex: bert-base-uncased.
    --dataset_name                  The name of which dataset you want to use to train or
                                    your model. ex: squad for using SQuAD.
    --num_train_epochs              Paramater to control how many training epochs you wish
                                    your model to train.
    --do_train                      Boolean denoting if the model should be trained
                                    or not. Default is false.
    --do_eval                       Boolean denoting if the model should be evaluated
                                    or not. Default is false.
    --per_device_train_batch_size   Size of each training batch based on samples per GPU.
                                    12 will fit in a 11gb GPU, 16 in a 16gb.
    --per_device_eval_batch_size    Size of each training batch based on samples per GPU.
                                    12 will fit in a 11gb GPU, 16 in a 16gb.
    --learning_rate                 Learning rate initial float value. ex: 3e-5.
    --max_seq_length                Int for the max sequence length to be parsed as a context
                                    window. ex: 384 tokens.
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

##########
Example command for training a 95% sparse BERT SQUAD model for 1 epoch with a unpruned teacher:
python examples/transformers/run_distill_qa.py \
    --teacher_model_name_or_path spacemanidol/neuralmagic-bert-squad-12layer-0sparse
    --student_model_name_or_path bert-base-uncased \
    --dataset_name squad \
    --num_train_epochs 1 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 12 \
    --learning_rate 3e-5 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir 95sparsity1epoch/ \
    --overwrite_output_dir \
    --cache_dir cache \
    --preprocessing_num_workers 8 \
    --seed 42 \
    --nm_prune_config prune_config_files/95sparsity1epoch.yaml \
    --do_onnx_export \
    --onnx_export_path 95sparsity1epoch/ \
    --distill_hardness 0.5 \
    --temperature 2.0 \
    --layers_to_keep 12 \
"""
import collections
import json
import logging
import math
import os
import random
import re
import sys

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import copy
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Module, Parameter
from torch.optim import Adam
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm
import wandb

import datasets
import transformers
from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_datasets_available,
    is_torch_tpu_available,
    set_seed,
)

from transformers.optimization import (
    Adafactor,
    AdamW,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from transformers.trainer_utils import PredictionOutput, is_main_process

from sparseml.pytorch.optim.manager import ScheduledModifierManager
from sparseml.pytorch.optim.optimizer import ScheduledOptimizer
from sparseml.pytorch.utils import ModuleExporter

from distill_trainer_qa import DistillQuestionAnsweringTrainer
from utils_qa import postprocess_qa_predictions


logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    teacher_model_name_or_path: Optional[str] = field(
        default="spacemanidol/neuralmagic-bert-squad-12layer-0sparse", metadata={"help": "Teacher model which needs to be a trained QA model"}
    )
    student_model_name_or_path: Optional[str] = field(
        default="bert-base-uncased", metadata={"help": "Student model"}
    )
    temperature: Optional[float] = field(
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
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    ####################################################################################
    # Start SparseML Integration
    ####################################################################################
    nm_prune_config: Optional[str] = field(
        default='recipes/noprune1epoch.yaml', metadata={"help": "The input file name for the Neural Magic pruning config"}
    )
    do_onnx_export: bool = field(
        default=False, metadata={"help": "Export model to onnx"}
    )
    onnx_export_path: Optional[str] = field(
        default='onnx-export', metadata={"help": "The filename and path which will be where onnx model is outputed"}
    )
    layers_to_keep: int = field(
        default=12, metadata={"help":"How many layers to keep for the model"}
    )
    ####################################################################################
    # End SparseML Integration
    ####################################################################################
    dataset_name: Optional[str] = field(
        default='squad', metadata={"help": "The name of the dataset to use (via the datasets library). The default is squad."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_query_length: int = field(
        default=30,
        metadata={
            "help": "The maximum total query length after tokenization"
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
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

####################################################################################
# Start SparseML Integration
####################################################################################
def load_optimizer(model, args):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer_cls = AdamW
    optimizer_kwargs = {
        "betas": (args.adam_beta1, args.adam_beta2),
        "eps": args.adam_epsilon,
    }
    optimizer_kwargs["lr"] = args.learning_rate
    return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)


def convert_example_to_features(example, tokenizer, max_seq_length, doc_stride, max_query_length):
    Feature = collections.namedtuple(
        "Feature",
        [
            "unique_id",
            "tokens",
            "example_index",
            "token_to_orig_map",
            "token_is_max_context",
        ],
    )
    extra = []
    unique_id = 0
    query_tokens = tokenizer.tokenize(example["question"])[0:max_query_length]
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example["context"]):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
    _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)
    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
            is_max_context = _check_is_max_context(
                doc_spans, doc_span_index, split_token_index
            )
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        feature = Feature(
            unique_id=unique_id,
            tokens=tokens,
            example_index=0,
            token_to_orig_map=token_to_orig_map,
            token_is_max_context=token_is_max_context,
        )
        extra.append(feature)
        unique_id += 1
        # extra is used as additional data but sparseml doesn't support it
    return (
        torch.from_numpy(np.array([np.array(input_ids, dtype=np.int64)])),
        torch.from_numpy(np.array([np.array(input_mask, dtype=np.int64)])),
        torch.from_numpy(np.array([np.array(segment_ids, dtype=np.int64)])),
    )


def _check_is_max_context(doc_spans, cur_span_index, position):
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index
    return cur_span_index == best_span_index

def drop_layers(model, layers_to_keep):
    layer_drop_matching = {
        1:[0],
        3:[0,5,11],
        6:[0,2,4,6,8,11],
        9:[0,2,3,4,5,7,8,9,11],
        12:[0,1,2,3,4,5,6,7,8,9,10,11],
    }
    encoder_layers = model.bert.encoder.layer # change based on model name
    assert layers_to_keep <= len(encoder_layers)
    assert layers_to_keep in layer_drop_matching.keys()
    trimmed_encoder_layers = nn.ModuleList()
    for i in layer_drop_matching[layers_to_keep]:
        trimmed_encoder_layers.append(encoder_layers[i])
    trimmed_model = copy.deepcopy(model)
    trimmed_model.bert.encoder.layer = trimmed_encoder_layers
    return trimmed_model

####################################################################################
# End SparseML Integration
####################################################################################

def main():
    ### Dataset processing classes in main due to hugging face custom dataset map
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=data_args.max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
        return tokenized_examples

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)
    # Post-processing:
    def post_processing_function(examples, features, predictions):
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=data_args.version_2_with_negative,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=data_args.null_score_diff_threshold,
            output_dir=training_args.output_dir,
            is_world_process_zero=trainer.is_world_process_zero(),
        )
        if data_args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
                for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [
                {"id": k, "prediction_text": v} for k, v in predictions.items()
            ]
        references = [
            {"id": ex["id"], "answers": ex[answer_column_name]}
            for ex in datasets["validation"]
        ]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=data_args.max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    transformers.utils.logging.set_verbosity_info()
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)
    if data_args.dataset_name is not None:
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files, field="data")
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.student_model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.student_model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
    )
    student_model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.student_model_name_or_path,
        from_tf=bool(".ckpt" in model_args.student_model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    teacher_model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.teacher_model_name_or_path,
        from_tf=bool(".ckpt" in model_args.teacher_model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    if data_args.layers_to_keep < len(student_model.bert.encoder.layer):
        logger.info("Keeping %s model layers", data_args.layers_to_keep)
        student_model = drop_layers(student_model, data_args.layers_to_keep)

    student_model_parameters = filter(lambda p: p.requires_grad, student_model.parameters())
    params = sum([np.prod(p.size()) for p in student_model_parameters])
    logger.info("Student Model has %s parameters", params)
    teacher_model_parameters = filter(lambda p: p.requires_grad, teacher_model.parameters())
    params = sum([np.prod(p.size()) for p in teacher_model_parameters])
    logger.info("Teacher Model has %s parameters", params)
    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    pad_on_right = tokenizer.padding_side == "right"

    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorWithPadding(tokenizer)
    )

    current_dir = os.path.sep.join(os.path.join(__file__).split(os.path.sep)[:-1])
    metric = load_metric(
        os.path.join(current_dir, "squad_v2_local")
        if data_args.version_2_with_negative
        else "squad"
    )

    if training_args.do_eval:
        validation_dataset = datasets["validation"].map(
            prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_train:
        train_dataset = datasets["train"].map(
            prepare_train_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    ####################################################################################
    # Start SparseML Integration
    ####################################################################################
    if training_args.do_train:
        optim = load_optimizer(student_model, training_args)
        steps_per_epoch = math.ceil(len(train_dataset) / (training_args.per_device_train_batch_size * training_args._n_gpu))
        manager = ScheduledModifierManager.from_yaml(data_args.nm_prune_config)
        training_args.num_train_epochs = float(manager.modifiers[0].end_epoch)
        optim = ScheduledOptimizer(optim, student_model, manager, steps_per_epoch=steps_per_epoch, loggers=None)
    ####################################################################################
    # End SparseML Integration
    ####################################################################################
    # Initialize our Trainer
    trainer = DistillQuestionAnsweringTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=validation_dataset if training_args.do_eval else None,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
        optimizers=(optim, None) if training_args.do_train else (None, None),
        teacher=teacher_model,
        distill_hardness = model_args.distill_hardness,
        temperature = model_args.temperature,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.student_model_name_or_path
            if os.path.isdir(model_args.student_model_name_or_path)
            else None
        )
        trainer.save_model()  # Saves the tokenizer too for easy upload

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        results = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in results.items():
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    ####################################################################################
    # Start SparseML Integration
    ####################################################################################
    if data_args.do_onnx_export:
        logger.info("*** Export to ONNX ***")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        exporter = ModuleExporter(
            student_model, output_dir='onnx-export'
        )
        sample_batch = convert_example_to_features(
            datasets["validation"][0],
            tokenizer,
            data_args.max_seq_length,
            data_args.doc_stride,
            data_args.max_query_length,
        )
        exporter.export_onnx(sample_batch=sample_batch, convert_qat=True)
    ####################################################################################
    # End SparseML Integration
    ####################################################################################

if __name__ == "__main__":
    main()
