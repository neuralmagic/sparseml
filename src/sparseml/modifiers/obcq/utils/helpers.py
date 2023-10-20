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

import logging
from math import ceil

import torch


_LOGGER = logging.getLogger(__name__)


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
            if isinstance(inp, tuple):
                inp = inp[0]
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
        if isinstance(inp, tuple):
            inp = inp[0]
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


def cache_attention_inputs(
    model, dataloader, device, nsamples, target_ids, layer_prefix
):
    if layer_prefix:
        embed_tokens = getattr(model.model, layer_prefix).embed_tokens
        first_layer = getattr(model.model, layer_prefix).layers[0]
    else:
        embed_tokens = model.model.embed_tokens
        first_layer = model.model.layers[0]
    embed_tokens.to(device)
    first_layer.to(device)
    cached_inputs = catch(
        model,
        first_layer,
        target_ids,  # ["attention_mask"],
        dataloader,
        nsamples,
    )
    embed_tokens.cpu()
    first_layer.cpu()
    torch.cuda.empty_cache()
    return cached_inputs


@torch.no_grad()
def ppl_eval_general(
    eval_logits, model, dataloader, dev, nsamples=None, max_samples_per_iteration=128
):
    _LOGGER.info("Evaluating perplexity...")

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
        logits = torch.cat(logits, dim=0).contiguous().to(torch.float32)

        labels = [sample[:, 1:].view(-1) for sample in samples]
        labels = torch.cat(labels, dim=0).to(dev)
        neg_log_likelihood += torch.nn.functional.cross_entropy(
            logits,
            labels,
            reduction="sum",
        )

        number_tokens += labels.numel()
        _LOGGER.info(torch.exp(neg_log_likelihood / number_tokens))

    ppl = torch.exp(neg_log_likelihood / number_tokens)
    _LOGGER.info(f"Perplexity: {ppl.item():3f}")

    return ppl.item()
