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

from math import ceil

import torch


class Catcher(torch.nn.Module):
    def __init__(self, module, target_keys):
        super().__init__()
        self.module = module
        self.cache = {key: [] for key in target_keys}
        self.target_keys = target_keys
        self.cache["inputs"] = []

    def forward(self, *args, **kwargs):
        self.cache["inputs"].append(args)
        for key in self.target_keys:
            self.cache[key].append(kwargs[key])
        raise ValueError

    def get_cache(self):
        return self.cache


def replace_module(model, old_module, new_module):
    for module_name, module in model.named_modules():
        if old_module == module:
            break

    current_module = model
    module_name = module_name.split(".")
    for child_module in module_name[:-1]:
        current_module = getattr(current_module, child_module)
    setattr(current_module, module_name[-1], new_module)


def catch(model, attention_layer, target_keys, data_loader, nsamples):
    catcher_module = Catcher(attention_layer, target_keys)
    replace_module(model, attention_layer, catcher_module)
    device = next(attention_layer.parameters()).device
    for input_id, inp in enumerate(data_loader):
        if nsamples is not None and input_id == nsamples:
            break
        try:
            model(inp.to(device), use_cache=False)
        except ValueError:
            pass
    replace_module(model, catcher_module, attention_layer)
    return catcher_module.get_cache()


def execute_offloaded_module(
    module,
    buffer,
    dev,
    nsamples=None,
    overwrite_buffer=True,
    cached_inputs=None,
    **kwargs,
):
    module.to(dev)
    if not overwrite_buffer:
        new_buffer = []
    for input_index, inp in enumerate(buffer):
        if nsamples is not None and input_index == nsamples:
            break
        if cached_inputs is None:
            module_kwargs = kwargs
        else:
            module_kwargs = {
                key: cached_inputs[key][input_index] for key in cached_inputs
            }
            module_kwargs.update(kwargs)
        output = module(inp.to(dev), **module_kwargs)
        if overwrite_buffer:
            buffer[input_index] = output
        else:
            new_buffer.append(output)

    module.cpu()
    torch.cuda.empty_cache()
    if overwrite_buffer:
        return buffer
    else:
        del buffer
        torch.cuda.empty_cache()
        return new_buffer


class OffLoadedModule(torch.nn.Module):
    def __init__(self, module, device):
        self._module = module
        self.device = device

        for name, child in module.named_modules():
            setattr(self._module, name, OffLoadedModule(child, device))

    def load_parameters(self):
        [p.to(self.device) for p in self._module.parameters(recurse=False)]

    def offload_parameters(self):
        [p.cpu() for p in self._module.parameters(recurse=False)]
        torch.cuda.empty_cache()

    def offloaded_forward(self, *args, **kwargs):
        self.load_parameters()
        output = self._module.forward(*args, **kwargs)
        self.offload_parameters()

        return output

    def forward(self, *args, **kwargs):
        return self.offloaded_forward(*args, **kwargs)


class SequentialCompressor(OffLoadedModule):
    def __init__(self, module, device, compression_algorithm=None, parent_module=None):
        self._module = module
        self.device = device
        self.compression_algorithm = compression_algorithm
        self.parent_module = parent_module
        self.cache = None
        self.cache_inputs = False

        for name, child in module.named_modules():
            setattr(
                self._module,
                name,
                SequentialCompressor(
                    child, device, compression_algorithm, self._module
                ),
            )

    def is_compressible(self):
        if self.compression_algorithm is not None:
            return self.compression_algorithm.is_compressible(self._module)
        else:
            return False

    def evaluate_cached_inputs(self, inputs, *args, **kwargs):
        if self.cache is None:
            if self.parent_module is None:
                for inp in inputs:
                    self._module(inp, *args, **kwargs)
            else:
                self.cache_inputs = True
                self.parent_module.evaluate_cached_inputs(inputs, *args, **kwargs)

    def clear_cached(self):
        del self.cache
        self.cache = None
        self.cache_inputs = False
        torch.cuda.empty_cache()

    def forward(self, *args, **kwargs):
        if self.cache_inputs:
            if self.cache is None:
                self.cache["args"] = [args]
                self.cache["kwargs"] = [kwargs]
            else:
                self.cache["args"].append(args)
                self.cache["kwargs"].append(kwargs)
        return self.offloaded_forward(*args, **kwargs)

    def compress(self, *args, **kwargs):
        if self.is_comporessible():
            self.cache_inputs = True
            self.parent_module.evaluate_cached_inputs(*args, **kwargs)
            self._module = self.compression_strategy(self._module, self.cached_inputs)
            self.clear_cache()
        else:
            for child in self._module.modules():
                child.compress(*args, **kwargs)


@torch.no_grad()
def ppl_eval_general(
    eval_logits,
    model,
    dataloader,
    dev,
    nsamples=None,
    max_samples_per_iteration=128,
):
    print("Evaluating perplexity...")

    if nsamples is None:
        nsamples = len(dataloader)

    number_iterations = int(ceil(nsamples / max_samples_per_iteration))
    neg_log_likelihood = 0.0
    number_tokens = 0
    for iteration in range(number_iterations):
        if iteration < number_iterations - 1:
            samples = dataloader[
                iteration
                * max_samples_per_iteration : (iteration + 1)
                * max_samples_per_iteration
            ]
        else:
            samples = dataloader[iteration * max_samples_per_iteration :]

        logits = eval_logits(model, samples, dev)

        vocabulary_size = logits[0].shape[-1]
        logits = [logit[:, :-1, :].view(-1, vocabulary_size) for logit in logits]
        logits = torch.concatenate(logits, dim=0).contiguous().to(torch.float32)

        labels = [sample[:, 1:].view(-1) for sample in samples]
        labels = torch.concatenate(labels, dim=0).to(dev)
        neg_log_likelihood += torch.nn.functional.cross_entropy(
            logits,
            labels,
            reduction="sum",
        )

        number_tokens += labels.numel()
        print(torch.exp(neg_log_likelihood / number_tokens), flush=True)

    ppl = torch.exp(neg_log_likelihood / number_tokens)
    print(f"Perplexity: {ppl.item():3f}")
    return ppl.item()


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
        i = random.randint(0, trainenc.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    testloader = [
        testenc[:, (i * seqlen) : ((i + 1) * seqlen)]
        for i in range(testenc.numel() // seqlen)
    ]
    testloader.append(testenc[:, (testenc.numel() // seqlen) * seqlen :])

    return trainloader, testloader, tokenizer


def get_ptb(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
    testdata = load_dataset("ptb_text_only", "penn_treebank", split="test")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")

    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc, tokenizer


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
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")["input_ids"]
            if trainenc.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc[:, i:j]
        trainloader.append(inp)

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]
    testloader = [
        valenc[:, (i * seqlen) : ((i + 1) * seqlen)]
        for i in range(valenc.numel() // seqlen)
    ]

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
        "prompt_input": "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response that appropriately completes the request."
        "\n\n### Instruction:\n{instruction}"
        "\n\n### Input:\n{input}"
        "\n\n### Response:\n",
        "prompt_no_input": "Below is an instruction that describes a task. "
        "Write a response that appropriately "
        "completes the request."
        "\n\n### Instruction:\n{instruction}"
        "\n\n### Response:\n",
    }

    from transformers import AutoTokenizer

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
