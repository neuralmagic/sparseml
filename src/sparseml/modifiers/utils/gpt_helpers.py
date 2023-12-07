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
import math
import time
from abc import ABC

import torch
import torch.nn as nn


try:
    import transformers
except ImportError as err:
    transformers = None
    transformers_err = err

__all__ = ["WandaGPT"]


DEBUG = False
_LOGGER = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPT(ABC):
    """
    Base class for pruning/quantization a single module
    with no sub-modules using information from input/output
    statistics

    Lifecycle:
        - add_batch
        - fasterprune
        - free
    """

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

    def free(self):
        """
        Free up memory used by the layer
        """
        raise NotImplementedError("Child class must implement `free`")


class LayerGPT(GPT):
    """
    Runs SparseGPT on a single module that contains no sub-modules

    Lifecycle:
        - add_batch
        - fasterprune
        - free


    :param layer: module to run SparseGPT on
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
        Run pruning and on the layer up to the target
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


class SparseGPT(LayerGPT):
    """
    Runs SparseGPT on a single module that contains no sub-modules

    Lifecycle:
        - add_batch
        - fasterprune
        - free


    :param layer: module to run SparseGPT on
    """

    def __init__(self, layer):
        super().__init__(layer=layer)
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor):
        """
        Add a batch of layer input and output data to the Hessian calculation

        :param inp: tensor containing layer input
        :param out: tensor containing layer our
        """
        self.store_inps_outs_for_debugging(inp, out)
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(
            self.layer, transformers.Conv1D
        ):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(
        self,
        sparsity: float,
        prunen: int = 0,
        prunem: int = 0,
        blocksize: int = 128,
        percdamp: float = 0.01,
    ):
        """
        Run pruning and quantization(if applicable) on the layer up to the target
        sparsity value.

        :param sparsity: target sparsity to reach for layer
        :param prunen: N for N:M pruning
        :param prunem: M for N:M pruning
        :param blocksize: Number of columns to compress in one pass
        :param percdamp: Amount of dampening to apply to H, as a fraction of the
            diagonal norm
        """
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        # See section 3.4 of https://arxiv.org/abs/2203.07259
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prunen == 0:
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1**2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    tmp = (
                        W1[:, i : (i + prunem)] ** 2
                        / (torch.diag(Hinv1)[i : (i + prunem)].reshape((1, -1))) ** 2
                    )
                    mask1.scatter_(
                        1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True
                    )

                q = w.clone()
                q[mask1[:, i]] = 0

                if hasattr(self.layer, "weight_fake_quant"):
                    scale = self.layer.weight_fake_quant.scale
                    zero_point = self.layer.weight_fake_quant.zero_point
                    dtype = self.layer.weight_fake_quant.dtype
                    qscheme = self.layer.weight_fake_quant.qscheme
                    if qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
                        q = torch.quantize_per_tensor(q, scale, zero_point, dtype)
                    else:
                        q = torch.quantize_per_channel(q, scale, zero_point, 0, dtype)
                    q = torch.dequantize(q)

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = W[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                _LOGGER.debug(torch.sum((self.layer(self._inp1) - self.out1) ** 2))
                _LOGGER.debug(torch.sum(Losses))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _LOGGER.info("time %.2f" % (time.time() - tick))
        _LOGGER.info("error %.2f" % torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if DEBUG:
            _LOGGER.debug(torch.sum((self.layer(self._inp1) - self.out1) ** 2))

    def free(self):
        """
        Free the Hessian memory after the layer is complete
        """
        self.H = None
        super().free()
