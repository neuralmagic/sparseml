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
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from sparseml.transformers.sparsification.obcq.sparsegpt import SparseGPT


_LOGGER = logging.getLogger(__name__)


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
    def __init__(self, model, layer, layer_index, inputs, manager, args):
        super().__init__(model=model)
        self.layer = layer
        self.layer_index = layer_index
        self.inputs = inputs
        self.manager = manager
        self.args = args

    def compressible_modules(self):
        quantize = self.args.get("quantize", False)
        if quantize:
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
        subset = self.compressible_modules()

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        # Run through the samples in order to compute Hessian matrix
        nsamples = self.inputs.shape[0]
        attention_mask = kwargs.get("attention_mask", None)
        for j in range(nsamples):
            attn_mask = (
                attention_mask[j]
                if isinstance(attention_mask, List)
                else attention_mask
            )
            self.layer(self.inputs[j].unsqueeze(0), attention_mask=attn_mask)[0]
        for h in handles:
            h.remove()

        return self.model, {"gpts": gpts}

    def compress(self, dev: str = "cuda:0", **kwargs):
        self.layer.to(dev)
        self.model, extras = self.pre_compress(**kwargs)

        gpts = extras["gpts"]
        for name in gpts:
            _LOGGER.info(f"Compressing {name}...")
            sparsity = self.args["sparsity"]
            gpts[name].fasterprune(
                sparsity,
                prunen=self.args["prunen"],
                prunem=self.args["prunem"],
                percdamp=self.args["percdamp"],
                blocksize=self.args["blocksize"],
            )
            gpts[name].free()

        self.model, extras = self.post_compress(**kwargs)

        return self.model, {"outputs": extras["outputs"]}

    def post_compress(self, **kwargs):
        outputs = torch.zeros_like(self.inputs)
        nsamples = self.inputs.shape[0]
        attention_mask = kwargs.get("attention_mask", None)
        for j in range(nsamples):
            attn_mask = (
                attention_mask[j]
                if isinstance(attention_mask, List)
                else attention_mask
            )
            outputs[j] = self.layer(
                self.inputs[j].unsqueeze(0), attention_mask=attn_mask
            )[0]

        return self.model, {"outputs": outputs}


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
