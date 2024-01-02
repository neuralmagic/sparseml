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
from typing import Dict

from torch.nn import Module

from sparseml.modifiers.obcq.utils.sgpt_wrapper import SGPTModuleWrapper
from sparseml.utils.pytorch import set_layer
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
    :param args: additional keyword arguments
    """

    def __init__(
        self, model: Module, layer: Module, layer_index: int, name: str, args: Dict
    ):
        self.model = model
        self.layer = layer
        self.layer_index = layer_index
        self.name = name
        self.args = args
        self.handles = None
        self.gpts = {}

    def compressible_modules(self) -> Dict:
        """
        Get the list of modules in the layer that can be compressed

        :return: dictionary of compressible modules
        """
        compressible_layers = get_prunable_layers(self.layer)
        return compressible_layers

    def pre_compress(self) -> Dict:
        """
        Sets up the SparseGPT objects for each compressible module, computes the Hessian
        for each using the calibration data.

        :return: SparseGPT objects for each module
        """
        subset = self.compressible_modules()

        for name in subset:
            layer = subset[name]
            wrapper = SGPTModuleWrapper(layer)
            full_name = ".".join(x for x in [self.name, name] if len(x) > 0)
            full_name = full_name.replace("_fsdp_wrapped_module.", "")
            set_layer(full_name, wrapper, self.model)
            self.gpts[name] = wrapper

        def add_batch(name):
            def tmp(_, inp, out):
                self.gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        self.handles = []
        for name in self.gpts:
            self.handles.append(subset[name].register_forward_hook(add_batch(name)))

    def post_compress(self):
        for handle in self.handles:
            handle.remove()

    def revert_layer_wrappers(self):
        for name, gpt_wrapper in self.gpts.items():
            full_name = ".".join(x for x in [self.name, name] if len(x) > 0)
            full_name = full_name.replace("_fsdp_wrapped_module.", "")
            set_layer(full_name, gpt_wrapper.layer, self.model)

        self.gpts = None

    def compress(self) -> Dict:
        """
        Run SparseGPT compression on all compressible modules in the layer
        """
        for name in self.gpts:
            _LOGGER.info(f"Compressing {name}...")
            sparsity = self.args["sparsity"]
            self.gpts[name].fasterprune(
                sparsity,
                prunen=self.args["prunen"],
                prunem=self.args["prunem"],
                percdamp=self.args["percdamp"],
                blocksize=self.args["blocksize"],
            )
