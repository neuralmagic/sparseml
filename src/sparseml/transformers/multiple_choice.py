#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning the library models for multiple choice.
"""
# You can also adapt this script on your own multiple choice task.
# Pointers for this are left as comments.

import logging
import os
import re
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, Union

import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    default_data_collator,
    set_seed,
)
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version

from sparseml.pytorch.utils.distributed import record
from sparseml.transformers.sparsification import Trainer, TrainingArguments
from sparseml.transformers.utils import SparseAutoModel, get_shared_tokenizer_src


# Will error if the minimal version of Transformers is not installed.
# Remove at your own risks.
check_min_version("4.18.0.dev0")


def prepare_swag(examples):
    """
    A data sample:
        video-id: anetv_dm5WXFiQZUQ
        fold-ind: 18419
        startphrase: He rides the motorcycle down the hall and into the
            elevator. He
        sent1: He rides the motorcycle down the hall and into the elevator.
        sent2: He
        gold-source: gold
        ending0: looks at a mirror in the mirror as he watches someone walk
            through a door.
        ending1: stops, listening to a cup of coffee with the seated woman,
            who's standing.
        ending2: exits the building and rides the motorcycle into a casino
            where he performs several tricks as people watch.
        ending3: pulls the bag out of his pocket and hands it to someone's
            grandma.
        label: 2
    Preprocessing to:
        1st_sentence = sent1
        2nd_sentence = sent2 + " " + ending[i]
    """
    ans_endings = ["ending0", "ending1", "ending2", "ending3"]
    num_choices = len(ans_endings)

    first_sentences = [[ctx] * num_choices for ctx in examples["sent1"]]
    second_sentences = [
        [f"{start} {examples[end][i]}" for end in ans_endings]
        for i, start in enumerate(examples["sent2"])
    ]

    labels = {}  # no need to modify existing labels
    return first_sentences, second_sentences, num_choices, labels


def prepare_csqa(examples):
    """
    A data sample:
        id: 075e483d21c29a511267ef62bedc0461
        question_concept: punishing
        question: The sanctions against the school were a punishing blow, and
            they seemed to what the efforts the school had made to change?
        choices:
            label: [A, B, C, D, E]
            text: [ignore, enforce, authoritarian, yell at, avoid]
        answerKey: A
    Preprocessing to:
      1st_sentence = Q: + question
      2nd_sentence = A: + choice[i]
    """
    answers = "choices"
    num_choices = len(examples[answers][0]["text"])
    prefix_Q = "Q: "
    prefix_A = "A: "

    first_sentences = [[prefix_Q + q] * num_choices for q in examples["question"]]
    second_sentences = [
        [prefix_A + ans for ans in answer["text"]] for answer in examples[answers]
    ]

    labels = {
        "label": [
            examples[answers][i]["label"].index(ans_key)
            for i, ans_key in enumerate(examples["answerKey"])
        ]
    }

    return first_sentences, second_sentences, num_choices, labels


def prepare_race(examples):
    """
    A data sample:
        example_id: middle4558.txt
        article: I planted a seed. Finally grow fruits. Today is a great day.
            Pick off the star for you. Pick off the moon for you. Let it rise
            for you every day. Become candles burning myself. Just light you
            up, hey!...
        answer: B
        question: Bae Seul-Ki   _   in the MV of the song according to the
            passage.
        options: [sang, danced, cried, laughed]
    Cleanup: replace newline char with space, remove multiple spaces
    Preprocessing to:
        1st_sentence = article
        2nd_sentence =
            a) question + fill_in_underscore with options[i]
            b) question + " " + options[i]
    """
    num_choices = len(examples["options"][0])
    first_sentences = []
    second_sentences = []
    labels_list = []

    for i in range(len(examples["example_id"])):
        # parse labels
        labels_list.append(ord(examples["answer"][i]) - ord("A"))

        # very simple cleanup heuristic, newlines and multiple spaces
        article = examples["article"][i]
        article = article.replace("\n", " ")
        article = re.sub(r"\s+", " ", article)
        first_sentences.append([article] * num_choices)

        qa_pairs = []
        q = examples["question"][i]
        # simple cleanup heuristic
        q = re.sub(r"\s+", " ", q)
        q = q.replace(" .", ".")
        q = q.replace(" ?", "?")
        if q.startswith(". "):
            q = q[2:]
        if q.endswith(".."):
            q = q[:-1]

        if "_" in q:  # fill-in task
            for ans in examples["options"][i]:
                # avoid doubling closing periods
                if ans.endswith(" ."):
                    ans = ans[:-2]
                if ans.endswith("."):
                    ans = ans[:-1]
                # handle multiple fill-ins in answer
                ans_qa = q
                for part_ans in ans.split(";"):
                    ans_qa = ans_qa.replace("_", part_ans, 1)
                qa_pairs.append(ans_qa)
        else:  # concat task
            qa_pairs = [
                q + " " + ans if ans.endswith(".") else q + " " + ans + "."
                for ans in examples["options"][i]
            ]

        qa_pairs = [re.sub(r"\s+", " ", qa) for qa in qa_pairs]
        qa_pairs = [qa.replace("\n", " ") for qa in qa_pairs]
        second_sentences.append(qa_pairs)

    labels = {"label": labels_list}
    return first_sentences, second_sentences, num_choices, labels


def prepare_winogrande(examples):
    """
    A data sample:
        sentence: The quilt Margot was making would take up the whole closet,
            because the _ was huge.
        option1: quilt
        option2: closet
        answer: 1
    Preprocessing to:
      1st_sentence = sentence up to _ (including filled-in _)
      2nd_sentence = remainder after _
    """
    num_choices = 2
    first_sentences = []
    second_sentences = []
    labels_list = []

    for i in range(len(examples["sentence"])):
        split = examples["sentence"][i].split("_")
        assert len(split) == 2  # more "_" must not happen
        first_sentences.append(
            [split[0] + examples["option1"][i], split[0] + examples["option2"][i]]
        )
        second_sentences.append([split[1], split[1]])
        labels_list.append(int(examples["answer"][i]) - 1)

    labels = {"label": labels_list}
    return first_sentences, second_sentences, num_choices, labels


def drop_samples_when_all_answers_identical(dataset_name, dataset):
    if dataset_name != "race":  # implement if needed for other tasks as well
        return dataset
    to_drop = []
    for i, sample in enumerate(dataset):
        if all(ans == sample["options"][0] for ans in sample["options"]):
            to_drop.append(i)
    _LOGGER.info(f"Dropped {len(to_drop)} samples because all answers are the same")
    return dataset.select(i for i in range(len(dataset)) if i not in to_drop)


_TASK_TO_KEYS = {
    "swag": {"prepare_data": prepare_swag},
    "commonsense_qa": {"prepare_data": prepare_csqa},
    "race": {"prepare_data": prepare_race},
    "winogrande": {
        "prepare_data": prepare_winogrande,
    },
}
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
    training and eval.

    Using `HfArgumentParser` we can turn this class into argparse
    arguments to be able to specify them on the command line
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the datasets library)."
            + " Supported: "
            + ",".join(_TASK_TO_KEYS.keys())
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The configuration name of the dataset to use "
                "(via the datasets library)."
            ),
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the "
            "perplexity on (a text file)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization."
            " If passed, sequences longer than this will be truncated,"
            " sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to the maximum sentence length."
            " If False, will pad the samples dynamically when batching to the"
            " maximum length in the batch. Efficient on GPU, very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the"
            " number of training examples to this value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the"
            " number of evaluation examples to this value if set."
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

    def __post_init__(self):
        if self.dataset_name is not None:
            self.dataset_name = self.dataset_name.lower()
            if self.dataset_name not in _TASK_TO_KEYS.keys():
                raise ValueError(
                    "Unknown dataset, pick one in " + ",".join(_TASK_TO_KEYS.keys())
                )
        elif self.train_file is None or self.validation_file is None:
            raise ValueError(
                "Need either a dataset_name or train_file and validation_file"
            )
        else:
            extension = self.train_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
            ], "`train_file` should be a csv or a json file."
            extension = self.validation_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
            ], "`validation_file` should be a csv or a json file."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from"
            " huggingface.co/models"
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
        metadata={
            "help": "Where do you want to store the pretrained models downloaded"
            " from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the"
            " tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name,"
            " tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running"
            " `transformers-cli login` (necessary to use this script with"
            " private models)."
        },
    )


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~file_utils.PaddingStrategy`], *optional*,
            defaults to `True`):
            Select a strategy to pad the returned sequences among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch
                (or no padding if only a single sequence if provided).
            - `'max_length'`: Pad to a max length specified with the argument
                `max_length` or to the maximum acceptable input length for the
                model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output
                a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on
            NVIDIA hardware with compute capability >= 7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)]
            for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


@record
def main():
    # See all possible arguments in
    # src/sparseml/transformers/sparsification/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already "
                "exists and is not empty. Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            _LOGGER.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To "
                "avoid this behavior, change the `--output_dir` or add "
                "`--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training
    # and evaluation files (see below) or just provide the name of one of the
    # public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    # For CSV/JSON files, this script will use the column called 'text' or the
    # first column if no column called 'text' is found.
    # You can easily tweak this behavior (see below).

    # In distributed training, the load_dataset function guarantee that only one
    # local process can concurrently download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension, data_files=data_files, cache_dir=model_args.cache_dir
        )
    # See more about loading any type of standard or custom dataset (from files,
    # python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

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

    model, teacher = SparseAutoModel.multiple_choice_from_pretrained_distil(
        model_name_or_path=(
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path
        ),
        model_kwargs={
            "config": config,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "use_auth_token": True if model_args.use_auth_token else None,
        },
        teacher_name_or_path=training_args.distill_teacher,
        teacher_kwargs={
            "cache_dir": model_args.cache_dir,
            "use_auth_token": True if model_args.use_auth_token else None,
        },
    )

    tokenizer_src = (
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else get_shared_tokenizer_src(model, teacher)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_src,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            _LOGGER.warning(
                "The picked tokenizer seems to have large `model_max_length`"
                f" ({tokenizer.model_max_length}). Picking 1024 instead. You can"
                " change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            _LOGGER.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is"
                " larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using"
                f" max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Preprocessing the datasets.
    def preprocess_function(examples):
        first_sentences, second_sentences, num_choices, labels = _TASK_TO_KEYS[
            data_args.dataset_name
        ]["prepare_data"](examples)

        # Flatten
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            truncation="only_first",
            max_length=max_seq_length,
            padding="max_length" if data_args.pad_to_max_length else False,
            return_overflowing_tokens=False,
        )

        # Un-flatten
        tokenized_dict = {
            k: [v[i : i + num_choices] for i in range(0, len(v), num_choices)]
            for k, v in tokenized_examples.items()
        }
        tokenized_dict.update(labels)
        return tokenized_dict

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = drop_samples_when_all_answers_identical(
            data_args.dataset_name, train_dataset
        )
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        eval_dataset = drop_samples_when_all_answers_identical(
            data_args.dataset_name, eval_dataset
        )
        with training_args.main_process_first(
            desc="validation dataset map pre-processing"
        ):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )

    # Data collator
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorForMultipleChoice(
            tokenizer=tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        )
    )

    # Metric
    def compute_metrics(eval_predictions):
        predictions, label_ids = eval_predictions
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        model_state_path=model_args.model_name_or_path,
        recipe=training_args.recipe,
        metadata_args=metadata_args,
        recipe_args=training_args.recipe_args,
        teacher=teacher,
        args=training_args,
        data_args=data_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
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

    # Exporting samples
    if data_args.num_export_samples > 0:
        trainer.save_sample_inputs_outputs(
            num_samples_to_export=data_args.num_export_samples
        )


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
