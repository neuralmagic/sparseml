import numpy as np
import torch
import torch.nn as nn

from layer_compressor import BaseCompressor, LayerCompressor
from model_preprocessor import (
    QuantizationModelPreprocessor,
    SmoothQuantModelPreprocessor,
)
from sequential import SequentialSparseGPT


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
        model = self.model
        layers = model.model.decoder.layers
        nsamples = len(dataloader) if nsamples is None else nsamples

        use_cache = model.config.use_cache
        model.config.use_cache = False

        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            dev
        )
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
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
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
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


class OPTDecoderLayerCompressor(LayerCompressor):
    ...


class OPTHeadCompressor(BaseCompressor):
    ...


def prepare_sparsegpt(model, dataloader, args, **kwargs) -> SequentialSparseGPT:
    model_preprocessors = []
    if args.smoothquant and args.smooth_activation_file is not None:
        model_preprocessors.append(
            SmoothQuantModelPreprocessor(args.smooth_activation_file)
        )
    if args.recipe:
        model_preprocessors.append(
            QuantizationModelPreprocessor(
                model, args.recipe, dataloader, args.observer_batches
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
    import torch

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
    name = kwargs.get("dataset", args.dataset)
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
