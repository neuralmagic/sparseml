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

import torch

from sparseml.experimental.sparsegpt.layer_compressor import BaseCompressor
from sparseml.experimental.sparsegpt.model_preprocessor import QuantizationModelPreprocessor
from sparseml.experimental.sparsegpt.sequential import SequentialSparseGPT
from sparseml.experimental.sparsegpt.utils import catch, execute_offloaded_module, ppl_eval_general


class SequentialSparseGPT_Llama2(SequentialSparseGPT):
    def compressible_layers(self):
        return self.model.model.layers


class Llama2BottomCompressor(BaseCompressor):
    """
    Llama2 specific
    """

    def compress(
        self,
        dataloader=None,
        nsamples: int = None,
        dev: str = "cuda:0",
        **kwargs,
    ):
        cached_inputs = cache_attention_inputs(self.model, dataloader, dev, nsamples)

        outputs = cached_inputs.pop("inputs")
        outputs = [o[0] for o in outputs]
        cached_inputs.update({"outputs": outputs})
        return self.model, cached_inputs


def prepare_sparsegpt(model, dataloader, args, dev) -> SequentialSparseGPT:
    model_preprocessors = []
    if args.recipe:
        model_preprocessors.append(
            QuantizationModelPreprocessor(
                model,
                args.recipe,
                dataloader,
                args.observer_batches,
                llama2_forward,
            )
        )
    bottom_compressor = Llama2BottomCompressor(model)
    sequential_sparsegpt = SequentialSparseGPT_Llama2(
        model,
        recipe=args.recipe,
        model_preprocessors=model_preprocessors,
        bottom_compressor=bottom_compressor,
        args=args,
    )

    return sequential_sparsegpt


def load_model(args):
    model = args.model

    from transformers import LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(model, torch_dtype="auto")
    model.eval()
    seqlen = model.config.max_position_embeddings
    return model, seqlen


def load_data(args, seqlen, split=0.1):
    name = args.dataset
    nsamples = args.nsamples
    model = args.model
    seed = args.seed

    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    elif "platypus" in name:
        return get_openplatypus(nsamples, seed, seqlen, model, split)


def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")["input_ids"]
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")["input_ids"]

    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    testloader = [testenc[:, (i*seqlen):((i+1)*seqlen)] for i in range(testenc.numel() // seqlen)]
    testloader.append(testenc[:, (testenc.numel() // seqlen)*seqlen:])

    return trainloader, testloader, tokenizer


def get_openplatypus(nsamples, seed, seqlen, model, split):
    from datasets import load_dataset

    traindata = load_dataset("garage-bAInd/Open-Platypus", split="train")

    import random

    random.seed(seed)
    traindata = list(traindata)
    random.shuffle(traindata)
    number_test_samples = max(1, int(split * len(traindata)))
    testdata = traindata[-number_test_samples:]
    traindata = traindata[:-number_test_samples]
    if nsamples is not None and nsamples < len(traindata):
        traindata = traindata[:nsamples]

    alpaca_template = {
        "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
        "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    }

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)

    def _process_sample(sample):
        if "input" in sample:
            processed_sample = alpaca_template["prompt_input"].format(instruction=sample["instruction"], input=sample["input"])
        else:
            processed_sample = alpaca_template["prompt_no_input"].format(instruction=sample["instruction"])

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


def cache_attention_inputs(model, data_loader, device, nsamples):
    model.model.embed_tokens.to(device)
    model.model.layers[0].to(device)
    cached_inputs = catch(model, model.model.layers[0], ["attention_mask", "position_ids"], data_loader, nsamples)
    model.model.embed_tokens.cpu()
    model.model.layers[0].cpu()
    torch.cuda.empty_cache()
    return cached_inputs


def llama2_forward(model, data_loader, device, nsamples=None):
    # Catch attention mask
    cached_inputs = cache_attention_inputs(model, data_loader, device, nsamples)
    buffer = [b[0] for b in cached_inputs.pop("inputs")]
    for layer in model.model.layers:
        buffer = execute_offloaded_module(
            layer,
            buffer,
            device,
            cached_inputs=cached_inputs,
            use_cache=False,
        )
        buffer = [b[0] for b in buffer]

    del cached_inputs
    torch.cuda.empty_cache()

    buffer = execute_offloaded_module(
        model.model.norm,
        buffer,
        device,
    )
    logits = execute_offloaded_module(
        model.lm_head,
        buffer,
        device,
    )

    return logits





def get_c4(nsamples, seed, seqlen, model):
    from datasets import load_dataset

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

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
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

def ppl_eval(
        args,
        model,
        dataloader,
        dev,
        nsamples=None,
        max_samples_per_iteration=128,
):
    return ppl_eval_general(
        llama2_forward,
        model,
        dataloader,
        dev,
        nsamples,
        max_samples_per_iteration,
    )
