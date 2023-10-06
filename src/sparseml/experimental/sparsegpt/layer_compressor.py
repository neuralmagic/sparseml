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

import inspect
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from sparseml.experimental.sparsegpt.quant import WeightFakeQuantizer
from sparseml.experimental.sparsegpt.sparsegpt import SparseGPT


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
    def __init__(self, model, layer, layer_index, inputs, manager, args):
        super().__init__(model=model)
        self.layer = layer
        self.layer_index = layer_index
        self.inputs = inputs
        self.manager = manager
        self.args = args

    def compressible_modules(self, **kwargs):
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
        if not self.args.sequential_hessian_within_layer:
            subset = self.compressible_modules(**kwargs)

            gpts = {}
            for name in subset:
                gpts[name] = SparseGPT(subset[name])
                if self.manager is not None and self.manager.quantization_modifiers:
                    gpts[name].quantizer = WeightFakeQuantizer(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in gpts:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            # Run through the samples in order to compute Hessian matrix
            nsamples = len(self.inputs)
            forward_args_spec = inspect.getfullargspec(self.layer.__class__.forward)
            passed_in_args = [arg for arg in forward_args_spec.args if arg in kwargs]
            for j in range(nsamples):
                passed_in_kwargs = {}
                for arg in passed_in_args:
                    if isinstance(kwargs[arg], List):
                        passed_in_kwargs[arg] = kwargs[arg][j]
                    else:
                        passed_in_kwargs[arg] = kwargs[arg]
                self.layer(self.inputs[j], **passed_in_kwargs)
            for h in handles:
                h.remove()

            return self.model, {"gpts": gpts}
        else:
            return self.model, {}

    def compress(self, dev: str = "cuda:0", **kwargs):
        self.layer.to(dev)
        self.model, extras = self.pre_compress(**kwargs)
        if not self.args.sequential_hessian_within_layer:
            gpts = extras["gpts"]
            for name in gpts:
                print(f"Compressing {name}...")
                sparsity = self.args.sparsity
                gpts[name].fasterprune(
                    sparsity,
                    prunen=self.args.prunen,
                    prunem=self.args.prunem,
                    percdamp=self.args.percdamp,
                    blocksize=self.args.blocksize,
                )
                gpts[name].free()
        else:
            self._sequentially_compress(**kwargs)

        self.model, extras = self.post_compress(**kwargs)

        return self.model, {"outputs": extras["outputs"]}

    def post_compress(self, **kwargs):
        nsamples = len(self.inputs)
        outputs = []
        forward_args_spec = inspect.getfullargspec(self.layer.__class__.forward)
        passed_in_args = [arg for arg in forward_args_spec.args if arg in kwargs]
        for j in range(nsamples):
            passed_in_kwargs = {}
            for arg in passed_in_args:
                if isinstance(kwargs[arg], List):
                    passed_in_kwargs[arg] = kwargs[arg][j]
                else:
                    passed_in_kwargs[arg] = kwargs[arg]
            outputs.append(self.layer(self.inputs[j], **passed_in_kwargs)[0])

        self.inputs = None
        torch.cuda.empty_cache()
        return self.model, {"outputs": outputs}

    def _sequentially_compress(self, **kwargs):
        subset = self.compressible_modules(**kwargs)

        forward_args_spec = inspect.getfullargspec(self.layer.__class__.forward)
        passed_in_args = [arg for arg in forward_args_spec.args if arg in kwargs]

        passed_in_kwargs = {}
        for arg in passed_in_args:
            if isinstance(kwargs[arg], List):
                passed_in_kwargs[arg] = kwargs[arg][0]
            else:
                passed_in_kwargs[arg] = kwargs[arg]
        order = _find_dependency_order(
            self.layer, subset, self.inputs[0], **passed_in_kwargs
        )

        nsamples = len(self.inputs)
        for name in order:
            gpts = SparseGPT(subset[name])
            if self.manager is not None and self.manager.quantization_modifiers:
                gpts.quantizer = WeightFakeQuantizer(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts.add_batch(inp[0].data, out.data)

                return tmp

            handle = subset[name].register_forward_hook(add_batch(name))
            for j in range(nsamples):
                passed_in_kwargs = {}
                for arg in passed_in_args:
                    if isinstance(kwargs[arg], List):
                        passed_in_kwargs[arg] = kwargs[arg][j]
                    else:
                        passed_in_kwargs[arg] = kwargs[arg]
                self.layer(self.inputs[j], **passed_in_kwargs)
            handle.remove()

            print(f"Compressing module {name} of layer {self.layer_index}")
            gpts.fasterprune(
                self.args.sparsity,
                prunen=self.args.prunen,
                prunem=self.args.prunem,
                percdamp=self.args.percdamp,
                blocksize=self.args.blocksize,
            )
            gpts.free()


def _find_dependency_order(layer, subset, an_input, **kwargs):
    order = []

    def exe_input(name):
        def _exe_input(_, inp, out):
            if name in subset:
                order.append(name)

        return _exe_input

    handles = [subset[name].register_forward_hook(exe_input(name)) for name in subset]
    layer(an_input, **kwargs)
    for h in handles:
        h.remove()
    return order


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
