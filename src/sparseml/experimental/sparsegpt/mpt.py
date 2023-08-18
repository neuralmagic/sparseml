import contextlib
import warnings

import torch
import torch.nn as nn

from layer_compressor import BaseCompressor, LayerCompressor
from llmfoundry import (
    COMPOSER_MODEL_REGISTRY,
    build_finetuning_dataloader,
    build_text_denoising_dataloader,
)
from llmfoundry.data.text_data import build_text_dataloader
from llmfoundry.utils.builders import build_tokenizer
from model_preprocessor import QuantizationModelPreprocessor
from omegaconf import OmegaConf as om
from sequential import SequentialSparseGPT


class SequentialSparseGPT_MPT(SequentialSparseGPT):
    def compressible_layers(self):
        return self.model.model.transformer.blocks


class MPTBottomCompressor(BaseCompressor):
    def compress(self, dev: str = "cuda:0", **kwargs):
        NSAMPLES = kwargs["nsamples"]
        data_seq_len = kwargs["data_seq_len"]
        dataloader = kwargs["dataloader"]

        model = self.model
        layers = self.model.model.transformer.blocks

        use_cache = model.config.use_cache
        model.config.use_cache = False
        layers = model.model.transformer.blocks

        model.model.transformer.wte = model.model.transformer.wte.to(dev)
        layers[0] = layers[0].to(dev)

        dtype = next(iter(model.parameters())).dtype
        inps = torch.zeros(
            (NSAMPLES, data_seq_len, model.config.d_model), dtype=dtype, device=dev
        )
        cache = []

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[len(cache)] = inp
                cache.append(kwargs["attn_bias"])
                raise ValueError

        layers[0] = Catcher(layers[0])
        i = 0
        for batch in dataloader:
            try:
                tmp = {k: v.to(dev) for k, v in batch.items()}
                model(tmp)
            except ValueError:
                pass
            i += 1
            if i == NSAMPLES:
                break
        layers[0] = layers[0].module

        layers[0] = layers[0].cpu()
        model.model.transformer.wte = model.model.transformer.wte.cpu()
        torch.cuda.empty_cache()

        extras = kwargs.copy()
        extras.updates({"use_cache": use_cache, "outputs": inps, "attn_bias": cache})

        self.model = model
        return model, extras


class MPTDecoderLayerCompressor(LayerCompressor):
    def __init__(self, model, layer, layer_index, inputs, manager, **kwargs):
        super().__init__(model, layer, layer_index, inputs, manager, **kwargs)

        self.min_layer = kwargs["min_layer"]
        self.max_layer = kwargs["max_layer"]
        self.prune_only = kwargs["prune_only"]
        self.invert = kwargs["invert"]

    def compressible_modules(self):
        subset = super().compressible_layers()
        excluded = [
            name
            for name in subset
            if (
                not (
                    self.min_layer <= self.layer_index < self.max_layer
                    and self.prune_only in name
                )
            )
            == (not self.invert)
        ]
        return [name for name in subset if name not in excluded]


class MPTHeadCompressor(BaseCompressor):
    ...


def prepare_sparsegpt(model, dataloader, args, **kwargs) -> SequentialSparseGPT:
    # TODO: Check with Eldar on additional preprocessing (e.g., weight untying)
    model_preprocessors = []
    if args.recipe:
        model_preprocessors.append(
            QuantizationModelPreprocessor(
                args.recipe, dataloader, args.observer_batches
            )
        )
    bottom_compressor = MPTBottomCompressor(model)
    sequential_sparsegpt = SequentialSparseGPT_MPT(
        model,
        recipe=args.recipe,
        model_preprocessors=model_preprocessors,
        bottom_compressor=bottom_compressor,
    )

    return sequential_sparsegpt


def load_model(args):
    cfg = _build_cfg(args)
    tokenizer = build_tokenizer(cfg.tokenizer)

    print("Initializing model...")
    init_context = contextlib.nullcontext()
    cfg.model.init_device = "cpu"
    with init_context:
        model = build_composer_model(cfg.model, tokenizer)
    return model, {"cfg": cfg, "tokenizer": tokenizer}


def load_data(args):
    cfg = _build_cfg(args)
    tokenizer = build_tokenizer(cfg.tokenizer)
    train_loader = build_dataloader(
        cfg.train_loader,
        tokenizer,
        cfg.device_train_batch_size,
    )
    test_loader = build_dataloader(
        cfg.eval_loader, tokenizer, cfg.device_eval_batch_size
    )

    return train_loader, test_loader, tokenizer


def build_composer_model(model_cfg, tokenizer):
    warnings.filterwarnings(
        action="ignore",
        message="Torchmetrics v0.9 introduced a new argument class property",
    )
    if model_cfg.name not in COMPOSER_MODEL_REGISTRY:
        raise ValueError(f"Not sure how to build model with name={model_cfg.name}")
    return COMPOSER_MODEL_REGISTRY[model_cfg.name](model_cfg, tokenizer)


def _build_cfg(args):
    yaml_path = args.yaml_path
    args_list = args.args_list

    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    return cfg


def build_dataloader(cfg, tokenizer, device_batch_size):
    if cfg.name == "text":
        return build_text_dataloader(
            cfg,
            tokenizer,
            device_batch_size,
        )
    elif cfg.name == "text_denoising":
        return build_text_denoising_dataloader(
            cfg,
            tokenizer,
            device_batch_size,
        )
    elif cfg.name == "finetuning":
        return build_finetuning_dataloader(
            cfg,
            tokenizer,
            device_batch_size,
        )
    else:
        raise ValueError(f"Not sure how to build dataloader with config: {cfg}")
