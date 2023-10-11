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

import numpy as np
import torch

from sparseml.experimental.sparsegpt.layer_compressor import (
    BaseCompressor,
    LayerCompressor,
)
from sparseml.experimental.sparsegpt.model_preprocessor import (
    QuantizationModelPreprocessor,
)
from sparseml.experimental.sparsegpt.sequential import SequentialSparseGPT
from sparseml.experimental.sparsegpt.utils import (
    catch,
    execute_offloaded_module,
    get_c4,
    get_ptb,
    get_wikitext2,
    ppl_eval_general,
)


class SequentialSparseGPT_OPT(SequentialSparseGPT):
    def compressible_layers(self):
        return self.model.model.decoder.layers


class OPTBottomCompressor(BaseCompressor):
    """
    OPT specific
    """

    def compress(
        self, dataloader=None, nsamples: int = None, dev: str = "cuda:0", **kwargs
    ):

        cached_inputs = cache_attention_inputs(self.model, dataloader, dev, nsamples)

        outputs = cached_inputs.pop("inputs")
        outputs = [o[0] for o in outputs]
        cached_inputs.update({"outputs": outputs})
        return self.model, cached_inputs


class OPTDecoderLayerCompressor(LayerCompressor):
    ...


class OPTHeadCompressor(BaseCompressor):
    ...


def cache_attention_inputs(model, data_loader, device, nsamples):
    model.model.decoder.embed_tokens.to(device)
    model.model.decoder.embed_positions.to(device)
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out.to(device)
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in.to(device)

    model.model.decoder.layers[0].to(device)
    cached_inputs = catch(
        model, model.model.decoder.layers[0], ["attention_mask"], data_loader, nsamples
    )

    model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in.cpu()

    model.model.decoder.layers[0].cpu()
    torch.cuda.empty_cache()
    return cached_inputs


def opt_forward(model, data_loader, device, nsamples=None):
    # Catch attention mask
    cached_inputs = cache_attention_inputs(model, data_loader, device, nsamples)
    buffer = [b[0] for b in cached_inputs.pop("inputs")]
    for layer in model.model.decoder.layers:
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


def prepare_sparsegpt(model, dataloader, args, **kwargs) -> SequentialSparseGPT:
    model_preprocessors = []
    if args.recipe:
        model_preprocessors.append(
            QuantizationModelPreprocessor(
                model,
                args.recipe,
                dataloader,
                args.observer_batches,
                opt_forward,
            )
        )
    bottom_compressor = OPTBottomCompressor(model)
    sequential_sparsegpt = SequentialSparseGPT_OPT(
        model,
        recipe=args.recipe,
        model_preprocessors=model_preprocessors,
        bottom_compressor=bottom_compressor,
        args=args,
    )

    return sequential_sparsegpt


def load_model(args):
    model = args.model

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM

    model = OPTForCausalLM.from_pretrained(model, torch_dtype="auto")
    seqlen = model.config.max_position_embeddings
    model.seqlen = seqlen
    return model


def load_data(args, seqlen, split=0.1):
    name = args.dataset
    nsamples = args.nsamples
    model = args.model
    seed = args.seed

    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if "ptb" in name:
        return get_ptb(nsamples, seed, seqlen, model)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, model)


################################################
#
# Data ultilities
#
################################################


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def ppl_eval(
    args,
    model,
    dataloader,
    dev,
    nsamples=None,
    max_samples_per_iteration=128,
):
    return ppl_eval_general(
        opt_forward,
        model,
        dataloader,
        dev,
        nsamples,
        max_samples_per_iteration,
    )
