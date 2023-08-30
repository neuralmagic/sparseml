import numpy as np
import torch
import torch.nn as nn

from layer_compressor import BaseCompressor, LayerCompressor
from model_preprocessor import QuantizationModelPreprocessor
from sequential import SequentialSparseGPT


class SequentialSparseGPT_Llama2(SequentialSparseGPT):
    def compressible_layers(self):
        return self.model.model.layers


class Llama2BottomCompressor(BaseCompressor):
    """
    Llama2 specific
    """

    def post_compress(
        self, dataloader=None, nsamples: int = None, dev: str = "cuda:0", **kwargs
    ):
        model = self.model
        layers = model.model.layers
        nsamples = len(dataloader) if nsamples is None else nsamples

        use_cache = model.config.use_cache
        model.config.use_cache = False

        model.model.embed_tokens = model.model.decoder.embed_tokens.to(dev)

        layers[0] = layers[0].to(dev)

        dtype = next(iter(model.parameters())).dtype
        inps = torch.zeros(
            (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
        )
        cache = {"i": 0, "attention_mask": None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache["i"]] = inp
                cache["i"] += 1
                cache["attention_mask"] = kwargs["attention_mask"]
                raise ValueError

        layers[0] = Catcher(layers[0])
        for batch in dataloader:
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
        layers[0] = layers[0].module

        layers[0] = layers[0].cpu()
        model.model.embed_tokens = model.model.decoder.embed_tokens.cpu()

        torch.cuda.empty_cache()

        outs = torch.zeros_like(inps)
        attention_mask = cache["attention_mask"]

        extras = {
            "use_cache": use_cache,
            "outputs": outs,
            "attention_mask": attention_mask,
        }
        self.model = model
        return model, extras


def prepare_sparsegpt(model, dataloader, args) -> SequentialSparseGPT:
    model_preprocessors = []
    if args.recipe:
        model_preprocessors.append(
            QuantizationModelPreprocessor(
                args.recipe, dataloader, args.observer_batches
            )
        )
    bottom_compressor = Llama2BottomCompressor(model)
    sequential_sparsegpt = SequentialSparseGPT_Llama2(
        model,
        recipe=args.recipe,
        model_preprocessors=model_preprocessors,
        bottom_compressor=bottom_compressor,
    )

    return sequential_sparsegpt

def load_model(args):
    model = args.model

    from transformers import LlamaForCausalLM, LlamaTokenizer

    model = LlamaForCausalLM.from_pretrained(model, torch_dtype="auto")
    model.seqlen = model.config.max_position_embeddings
    return model

def load_data(args):
    name = args.dataset
    nsamples = args.nsamples
    seqlen = args.max_seq_len
    model = args.model
    seed = args.seed

    return get_wikitext2(nsamples, seed, seqlen, model)


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