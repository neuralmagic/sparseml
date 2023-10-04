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

from typing import List

import torch
from torch.nn import Module
from transformers import LlamaForCausalLM, OPTForCausalLM

from sparseml.modifiers.obcq.utils.utils import (
    cache_attention_inputs,
    execute_offloaded_module,
)


__all__ = ["load_opt_model", "load_llama_model", "opt_forward", "llama_forward"]


def load_opt_model(model_path: str) -> torch.nn.Module:
    """
    Load a pretrained OPT model from the specified hugging face path

    :param model_path: hugging face path to model
    :return: loaded pretrained model
    """

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    model = OPTForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    model.seqlen = model.config.max_position_embeddings
    return model


def load_llama_model(model_path: str) -> torch.nn.Module:
    """
    Load a pretrained Llama model from the specified hugging face path

    :param model_path: hugging face path to model
    :return: loaded pretrained model
    """
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    model.eval()
    model.seqlen = model.config.max_position_embeddings
    return model


def opt_forward(model: Module, data_loader: List, device: str, nsamples: int = None):
    """
    Run a forward pass of OPT, used for perplexity evaluation

    :param model: Pytorch module to run
    :param data_loader: data to run through model
    :param device: device name to perform computation on
    :param nsamples: number of samples of data_loader to run, None to run them all

    :return: logits output of the model
    """
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

    if model.model.decoder.final_layer_norm is not None:
        buffer = execute_offloaded_module(
            model.model.decoder.final_layer_norm,
            buffer,
            device,
        )
    if model.model.decoder.project_out is not None:
        buffer = execute_offloaded_module(
            model.model.decoder.project_out,
            buffer,
            device,
        )
    logits = execute_offloaded_module(
        model.lm_head,
        buffer,
        device,
    )

    return logits


def llama_forward(model: Module, data_loader: List, device: str, nsamples: int = None):
    """
    Run a forward pass of Llama, used for perplexity evaluation

    :param model: Pytorch module to run
    :param data_loader: data to run through model
    :param device: device name to perform computation on
    :param nsamples: number of samples of data_loader to run, None to run them all

    :return: logits output of the model
    """
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
