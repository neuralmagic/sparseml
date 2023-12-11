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
import time

import torch
import torch.nn as nn
import transformers

from sparseml.modifiers.utils.gpt_helpers import LayerGPT

_LOGGER = logging.getLogger(__name__)
DEBUG = False


class WandaGPT(LayerGPT):
    def __init__(self, layer):
        super().__init__(layer=layer)
        self.scaler_row = torch.zeros((self.columns), device=self.dev)

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor):
        """
        Add a batch of layer input and output data to the layer
        statistics calculation

        :param inp: tensor containing layer input
        :param out: tensor containing layer output
        """
        self.store_inps_outs_for_debugging(inp, out)
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size
        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples

    def fasterprune(
        self,
        sparsity: float,
        prunen: int = 0,
        prunem: int = 0,
    ):
        """
        Run pruning on the layer up to the target
        sparsity value.

        :param sparsity: target sparsity to reach for layer
        :param prunen: N for N:M pruning
        :param prunem: M for N:M pruning
        """
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        W_metric = torch.abs(W) * torch.sqrt(self.scaler_row.reshape((1, -1)))

        # initialize a mask to be all False
        W_mask = torch.zeros_like(W_metric) == 1
        if prunen != 0:
            # structured n:m sparsity
            for ii in range(W_metric.shape[1]):
                if ii % prunem == 0:
                    tmp = W_metric[:, ii : (ii + prunem)].float()
                    W_mask.scatter_(
                        1,
                        ii + torch.topk(tmp, prunen, dim=1, largest=False)[1],
                        True,
                    )
        else:
            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            indices = sort_res[1][:, : int(W_metric.shape[1] * sparsity)]
            W_mask.scatter_(1, indices, True)

        W[W_mask] = 0  # set weights to zero

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _LOGGER.info("time %.2f" % (time.time() - tick))

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if DEBUG:
            _LOGGER.debug(torch.sum((self.layer(self._inp1) - self.out1) ** 2))

    def free(self):
        """
        Free memory after the layer is complete
        """
        self.scaler_row = None
        super().free()
