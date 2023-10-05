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

import random
from typing import List, Tuple

import torch
from datasets import load_dataset
from torch.nn import Module
from transformers import AutoTokenizer, GPT2Tokenizer


__all__ = ["get_wikitext2", "get_ptb", "get_c4", "get_openplatypus"]


def get_wikitext2(
    nsamples: int, seed: int, seqlen: int, model: Module
) -> Tuple[List, GPT2Tokenizer]:
    """
    load nsamples of tokenized data from the wikitext2 dataset of length seqlen

    :param nsamples: number of samples to load
    :param seed: seed to use for selecting random samples from dataset
    :param seqlen: sequence length of each sample
    :param model: trained pytorch module to load tokenizer from
    :return: list of random samples from wikitext and tokenizer
    """
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt").input_ids
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt").input_ids

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        start = random.randint(0, trainenc.shape[1] - seqlen - 1)
        end = start + seqlen
        inp = trainenc[:, start:end]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, testenc, tokenizer


def get_ptb(
    nsamples: int, seed: int, seqlen: int, model: Module
) -> Tuple[List, GPT2Tokenizer]:
    """
    load nsamples of tokenized data from the ptb dataset of length seqlen

    :param nsamples: number of samples to load
    :param seed: seed to use for selecting random samples from dataset
    :param seqlen: sequence length of each sample
    :param model: trained pytorch module to load tokenizer from
    :return: list of random samples from ptb and tokenizer
    """
    traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
    testdata = load_dataset("ptb_text_only", "penn_treebank", split="test")
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        start = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        end = start + seqlen
        inp = trainenc.input_ids[:, start:end]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc, tokenizer


def get_c4(
    nsamples: int, seed: int, seqlen: int, model: Module
) -> Tuple[List, GPT2Tokenizer]:
    """
    load nsamples of tokenized data from the c4 dataset of length seqlen

    :param nsamples: number of samples to load
    :param seed: seed to use for selecting random samples from dataset
    :param seqlen: sequence length of each sample
    :param model: trained pytorch module to load tokenizer from
    :return: list of random samples from c4 and tokenizer
    """
    traindata = load_dataset(
        "allenai/c4",
        "allenai--c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    valdata = load_dataset(
        "allenai/c4",
        "allenai--c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            start_idx = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[start_idx]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        start = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        end = start + seqlen
        inp = trainenc.input_ids[:, start:end]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc, tokenizer


def get_openplatypus(
    nsamples: int, seed: int, seqlen: int, model: Module, split: float = 0.1
) -> Tuple[List, GPT2Tokenizer]:
    """
    load nsamples of tokenized data from the wikitext2 dataset of length seqlen

    :param nsamples: number of samples to load
    :param seed: seed to use for selecting random samples from dataset
    :param seqlen: sequence length of each sample
    :param model: trained pytorch module to load tokenizer from
    :param split: train/test split
    :return: list of random samples from wikitext and tokenizer
    """
    traindata = load_dataset("garage-bAInd/Open-Platypus", split="train")

    random.seed(seed)
    traindata = list(traindata)
    random.shuffle(traindata)
    number_test_samples = max(1, int(split * len(traindata)))
    testdata = traindata[-number_test_samples:]
    traindata = traindata[:-number_test_samples]
    if nsamples is not None and nsamples < len(traindata):
        traindata = traindata[:nsamples]

    alpaca_template = {
        "prompt_input": "Below is an instruction that describes a task, paired with an "
        "input that provides further context. Write a response that appropriately "
        "completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n"
        "{input}\n\n### Response:\n",
        "prompt_no_input": "Below is an instruction that describes a task. Write a "
        "response that appropriately completes the request.\n\n### Instruction:\n{"
        "instruction}\n\n### Response:\n",
    }

    tokenizer = AutoTokenizer.from_pretrained(model)

    def _process_sample(sample):
        if "input" in sample:
            processed_sample = alpaca_template["prompt_input"].format(
                instruction=sample["instruction"], input=sample["input"]
            )
        else:
            processed_sample = alpaca_template["prompt_no_input"].format(
                instruction=sample["instruction"]
            )

        if "output" in sample:
            processed_sample += sample["output"]

        tokenized_sample = tokenizer(
            processed_sample,
            truncation=True,
            max_length=seqlen,
            return_tensors="pt",
            padding=False,
        )["input_ids"][0]

        if tokenized_sample[-1] != tokenizer.eos_token_id:
            if len(tokenized_sample) == seqlen:
                tokenized_sample[-1] = tokenizer.eos_token_id
            else:
                tokenized_sample = torch.concatenate(
                    (tokenized_sample, torch.tensor((tokenizer.eos_token_id,))),
                )

        tokenized_sample = torch.unsqueeze(tokenized_sample, dim=0)

        return tokenized_sample

    trainenc = [_process_sample(sample) for sample in traindata]
    testenc = [_process_sample(sample) for sample in testdata]

    return trainenc, testenc, tokenizer
