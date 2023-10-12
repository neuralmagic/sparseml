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
from sparseml.experimental.sparsegpt.model_preprocessor import (
    QuantizationModelPreprocessor,
)
from sparseml.experimental.sparsegpt.sequential import SequentialSparseGPT
from sparseml.experimental.sparsegpt.utils import (
    catch,
    execute_offloaded_module,
    get_openplatypus,
    get_wikitext2,
    ppl_eval_general,
)


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


def cache_attention_inputs(model, data_loader, device, nsamples):
    model.model.embed_tokens.to(device)
    model.model.layers[0].to(device)
    cached_inputs = catch(
        model,
        model.model.layers[0],
        ["attention_mask", "position_ids"],
        data_loader,
        nsamples,
    )
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
