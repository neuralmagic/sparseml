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
import logging
from typing import Dict, List

import torch
from torch.nn import Module

from sparseml.modifiers.obcq.utils.sparsegpt import SparseGPT
from sparseml.pytorch.utils.helpers import get_dependency_order
from sparseml.utils.pytorch.module import get_prunable_layers


_LOGGER = logging.getLogger(__name__)


class LayerCompressor:
    """
    Runs the SparseGPT algorithm on a single layer using calibration data inputs

    Lifecycle:
        - compress
            - pre_compress_parallel (optional)
            - add_batch
            - fasterprune
            - post_compress

    :param model: model containing the layer we are running compression on
    :param layer: layer to run compression on
    :param layer_index: index of layer in the model
    :param inputs: calibration data to pass through the layer
    :param args: additional keyword arguments
    """

    def __init__(
        self, model: Module, layer: Module, layer_index: int, inputs: List, args: Dict
    ):
        self.model = model
        self.layer = layer
        self.layer_index = layer_index
        self.inputs = inputs
        self.args = args

    def compressible_modules(self) -> Dict:
        """
        Get the list of modules in the layer that can be compressed

        :return: dictionary of compressible modules
        """
        compressible_layers = get_prunable_layers(self.layer)
        return compressible_layers

    def pre_compress_parallel(self, **kwargs) -> Dict:
        """
        Sets up the SparseGPT objects for each compressible module, computes the Hessian
        for each using the calibration data.

        :return: SparseGPT objects for each module
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

        # Run through the samples in order to compute Hessian matrix for each module
        nsamples = len(self.inputs)
        forward_args_spec = inspect.getfullargspec(self.layer.__class__.forward)
        passed_in_args = [arg for arg in forward_args_spec.args if arg in kwargs]
        for sample_idx in range(nsamples):
            passed_in_kwargs = {}
            for arg in passed_in_args:
                if isinstance(kwargs[arg], List):
                    passed_in_kwargs[arg] = kwargs[arg][sample_idx]
                else:
                    passed_in_kwargs[arg] = kwargs[arg]
            self.layer(self.inputs[sample_idx], **passed_in_kwargs)
        for h in handles:
            h.remove()

        return {"gpts": gpts}

    def compress(self, dev: str = "cuda:0", **kwargs) -> Dict:
        """
        Run SparseGPT compression on all compressible modules in the layer

        :param dev: device to run computation on
        """
        self.layer.to(dev)
        if not self.args["sequential_update"]:
            # compute Hessians ahead of time
            extras = self.pre_compress_parallel(**kwargs)
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
        else:
            # Hessians computed layer by layer
            self.sequentially_compress(**kwargs)

        extras = self.post_compress(**kwargs)

        return {"outputs": extras["outputs"]}

    def post_compress(self, **kwargs) -> Dict:
        """
        Clean up after compression

        :return: outputs of the layer
        """
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
        # once we've finished compressing the layer, move it back to CPU to save memory
        self.layer.to("cpu")
        torch.cuda.empty_cache()

        return {"outputs": outputs}

    def sequentially_compress(self, **kwargs):
        """
        Run compression module by module, in dependency order. Unlike in parallel
        compression, we compute the Hessians layer by layer instead of computing them
        all up front. This saves on memory and means compression in earlier layers
        affects the inputs to later layers
        """
        subset = self.compressible_modules()

        # filter kwargs that are expected as layer inputs
        forward_args_spec = inspect.getfullargspec(self.layer.__class__.forward)
        passed_in_args = [arg for arg in forward_args_spec.args if arg in kwargs]

        passed_in_kwargs = {}
        for arg in passed_in_args:
            if isinstance(kwargs[arg], List):  # take the first batch
                passed_in_kwargs[arg] = kwargs[arg][0]
            else:
                passed_in_kwargs[arg] = kwargs[arg]
        order = get_dependency_order(
            self.layer, subset, self.inputs[0], **passed_in_kwargs
        )

        nsamples = len(self.inputs)
        for name in order:  # create SparseGPT object for each compressible module
            gpts = SparseGPT(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts.add_batch(inp[0].data, out.data)

                return tmp

            # add SparseGPT hook for current module
            handle = subset[name].register_forward_hook(add_batch(name))
            for sample_idx in range(nsamples):
                passed_in_kwargs = {}
                for arg in passed_in_args:
                    if isinstance(kwargs[arg], List):
                        passed_in_kwargs[arg] = kwargs[arg][sample_idx]
                    else:
                        passed_in_kwargs[arg] = kwargs[arg]
                # run layer, triggering SparseGPT add_batch for current module
                self.layer(self.inputs[sample_idx], **passed_in_kwargs)
            handle.remove()

            _LOGGER.info(f"Compressing module {name} of layer {self.layer_index}")
            gpts.fasterprune(  # run SparseGPT algorithm on current module
                self.args["sparsity"],
                prunen=self.args["prunen"],
                prunem=self.args["prunem"],
                percdamp=self.args["percdamp"],
                blocksize=self.args["blocksize"],
            )
            gpts.free()
