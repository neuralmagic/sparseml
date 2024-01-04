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

from abc import ABC

import torch
import torch.nn as nn


try:
    import transformers
except ImportError as err:
    transformers = None
    transformers_err = err

__all__ = ["ModuleCompressor"]


DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class ModuleCompressor(ABC):
    """
    Base Abstract class for pruning/quantization a single module
    with no sub-modules using information from input/output
    statistics

    :param layer: module to run compression on
    """

    def __init__(self, layer):
        if transformers is None:
            raise transformers_err

        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.nsamples = 0

    def store_inps_outs_for_debugging(self, inp, out):
        if DEBUG:
            self._inp1 = inp
            self.out1 = out

    def free(self):
        """
        Free memory after the layer is complete
        calls torch.cuda.empty_cache() to defragement GPU memory
        """
        if DEBUG:
            if hasattr(self, "_inp1"):
                self._inp1 = None
            if hasattr(self, "out1"):
                self.out1 = None
        torch.cuda.empty_cache()

    def add_batch(self, *args, **kwargs):
        """
        Add a batch of layer input and output data to the layer
        statistics calculation
        """
        raise NotImplementedError("Child class must implement `add_batch`")

    def fasterprune(self, *args, **kwargs):
        """
        Run pruning and on the layer up to the target
        sparsity
        """
        raise NotImplementedError("Child class must implement `fasterprune`")
