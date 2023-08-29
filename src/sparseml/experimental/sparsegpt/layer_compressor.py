from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from quant import Quantizer, WeightFakeQuantizer
from sparsegpt import SparseGPT


DEFAULT_WBITS = 16


class BaseCompressor:
    def __init__(self, model):
        self.model = model

    def pre_compress(self, dev: str = "cuda:0", **kwargs) -> Tuple[nn.Module, Dict]:
        return self.model, {}

    def compress(self, dev: str = "cuda:0", **kwargs) -> Tuple[nn.Module, Dict]:
        return self.model, {}

    def post_compress(self, dev: str = "cuda:0", **kwargs) -> Tuple[nn.Module, Dict]:
        return self.model, {}


class LayerCompressor(BaseCompressor):
    def __init__(self, model, layer, layer_index, inputs, manager, **kwargs):
        super().__init__(model=model)
        self.layer = layer
        self.layer_index = layer_index
        self.inputs = inputs
        self.manager = manager

        self.attention_mask = kwargs.get("attention_mask", None)
        self.wbits = kwargs.get("wbits", DEFAULT_WBITS)
        self.perchannel = kwargs.get("perchannel", False)
        self.sparsity = kwargs["sparsity"]
        self.prunen = kwargs["prunen"]
        self.prunem = kwargs["prunem"]
        self.percdamp = kwargs["percdamp"]
        self.blocksize = kwargs["blocksize"]

        self.gpts = {}

    def compressible_modules(self):
        if self.manager is not None and self.manager.quantization_modifiers:
            # The layer names are changed due to quantization modifiers, therefore
            # we need a slightly different func to retrieve layers
            modules = _find_quant_layers(self.layer)
        else:
            modules = _find_layers(self.layer)
        return modules

    def pre_compress(self, **kwargs):
        """
        Set up SparseGPT objects, compute Hessian
        """
        manager = self.manger
        subset = self.compressible_modules()

        gpts = self.gpts
        for name in subset:
            gpts[name] = SparseGPT(subset[name])
            if self.wbits < 16:
                if manager is not None and manager.quantization_modifiers:
                    gpts[name].quantizer = WeightFakeQuantizer(subset[name])
                else:
                    gpts[name].quantizer = Quantizer()
                    gpts[name].quantizer.configure(
                        self.wbits, perchannel=self.perchannel, sym=False, mse=False
                    )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        # Run through the samples in order to compute Hessian matrix
        nsamples = self.inputs.shape[0]
        for j in range(nsamples):
            attn_mask = (
                self.attention_mask[j]
                if isinstance(self.attention_mask, List)
                else self.attention_mask
            )
            self.layer(self.inputs[j].unsqueeze(0), attention_mask=attn_mask)[0]
        for h in handles:
            h.remove()

        return self.model, {}

    def compress(self, **kwargs):
        gpts = self.gpts
        for name in gpts:
            print(f"Compressing {name}...")
            sparsity = self.sparsity
            gpts[name].fasterprune(
                sparsity,
                prunen=self.prunen,
                prunem=self.prunem,
                percdamp=self.percdamp,
                blocksize=self.blocksize,
            )
            gpts[name].free()

        return self.model, kwargs

    def post_compress(self, model, **kwargs):
        outputs = torch.zeros_like(self.inputs)
        nsamples = self.inputs.shape[0]
        for j in range(nsamples):
            attn_mask = (
                self.attention_mask[j]
                if isinstance(self.attention_mask, List)
                else self.attention_mask
            )
            outputs[j] = self.layer(
                self.inputs[j].unsqueeze(0), attention_mask=attn_mask
            )[0]

        return self.model, kwargs.update({"outputs": outputs})


def _find_quant_layers(module, layers=[torch.nn.qat.Linear], name=""):
    if type(module) in layers:
        pieces = name.split(".")
        if pieces[-1] == "module":
            name = ".".join(pieces[:-1])
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            _find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def _find_layers(module, layers=[nn.Conv2d, nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            _find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res
