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
import torch.nn as nn

from sparseml.experimental.sparsegpt.layer_compressor import BaseCompressor, LayerCompressor
from sparseml.experimental.sparsegpt.model_preprocessor import QuantizationModelPreprocessor
from sparseml.experimental.sparsegpt.sequential import SequentialSparseGPT
from sparseml.experimental.sparsegpt.utils import catch, execute_offloaded_module, ppl_eval_general


smoothquant_subgraph_keys = [
    {
        "module_to_balance": ["q_proj", "k_proj", "v_proj"],
        "module_to_merge_scale": ["self_attn_layer_norm"]
    },
    {
        "module_to_balance": ["gate_proj", "up_proj"],
        "module_to_merge_scale": ["final_layer_norm"]
    },
    {
        "module_to_balance": ["down_proj"],
        "module_to_merge_scale": ["up_proj", "linear"],
    },
]

class QuantizationModelPreprocessor_OPT(QuantizationModelPreprocessor):
    def smoothquant_layers(self):
        return self.model.model.decoder.layers


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
    if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
    ):
        model.model.decoder.project_out.to(device)
    if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
    ):
        model.model.decoder.project_in.to(device)

    model.model.decode.layers[0].to(device)
    cached_inputs = catch(model, model.model.decoder.layers[0], ["attention_mask"], data_loader, nsamples)

    model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions.cpu()
    if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
    ):
        model.model.decoder.project_out.cpu()
    if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
    ):
        model.model.decoder.project_in.cpu()

    model.model.decode.layers[0].cpu()
    torch.cuda.empty_cache()
    return cached_inputs


def opt_forward(model, data_loader, device, nsamples=None):
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
            QuantizationModelPreprocessor_OPT(
                model,
                args.recipe,
                dataloader,
                args.observer_batches,
                opt_forward,
                smoothquant = args.smoothquant or args.logarithmic_equalization,
                smoothquant_kwargs = {
                    "subgraph_keys": smoothquant_subgraph_keys,
                    "alpha": args.smoothquant_alpha,
                    "logarithmic_equalization": args.logarithmic_equalization,
                },
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
    model.seqlen = model.config.max_position_embeddings
    return model


def load_data(args, **kwargs):
    name = kwargs.get("dataset", None)
    name = args.dataset if name is None else name
    nsamples = args.nsamples
    seqlen = args.data_sequence_length
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


def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

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
        opt_forward,
        model,
        dataloader,
        dev,
        nsamples,
        max_samples_per_iteration,
    )

