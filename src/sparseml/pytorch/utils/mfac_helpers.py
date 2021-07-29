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

"""
Helper functions for performing Matrix-Free Approximate Curvature (M-FAC)
pruning.
"""

import math
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.parallel.parallel_apply import parallel_apply

from sparseml.pytorch.utils import tensors_module_forward


__all__ = [
    "GradSampler",
    "MFACOptions",
    "FisherInverse",
    "FisherInverseFast",
    "FisherInverseFastBlock",
    "FisherInverseFastPageSwap",
    "compute_hessian_inv",
]


class GradSampler:
    """
    Class for computing gradient samples for a Model given a sample data loader and
    loss function.

    :param data_loader: iterator of data samples to use as model inputs and their loss
        targets. Samples can either be single tensors as model input or a list of
        inputs and should be iterated in tuples with their targets
    :param loss_fn: function to be called on model outputs to compute the loss at
        each step
    """

    def __init__(
        self,
        data_loader: Iterator[Tuple[Union[Tensor, List[Tensor]], Any]],
        loss_fn: Callable[[Tensor], Tensor],
    ):
        if not isinstance(data_loader, Iterable):
            raise ValueError(
                "data_loader for GradSampler must be Iterable, received object of "
                f"type {type(data_loader)}"
            )
        if not callable(loss_fn):
            raise ValueError(
                "loss_fn for GradSampler must be callable, given input "
                f"with type {type(loss_fn)}"
            )

        self._data_loader = data_loader
        self._loss_fn = loss_fn

    def module_forward(self, module: Module, data: Union[Tensor, List[Tensor]]) -> Any:
        """
        :param module: module to perform forward pass with
        :param data: single data sample to pass to module
        :return: output(s) of the module forward pass
        """
        if isinstance(data, Tensor):
            data = [data]

        return tensors_module_forward(*data, module)

    def module_backward(self, module_outputs: Any, targets: Any):
        """
        Computes module loss based on the given module outputs, target data and loss
        function

        :param module_outputs: outputs of a forward pass from a module
        :param targets: target outputs for the module to be used for the loss function
        """
        loss = self._loss_fn(module_outputs, targets)
        loss.backward()

    def iter_module_backwards(
        self, module: Module, num_grads: int
    ) -> Generator[int, None, None]:
        """

        :param module: module to compute gradients for
        :param num_grads: number of gradient samples to compute
        :return: generator that yields after every gradient is computed with the index
            of the gradient sample number
        """
        computed_grads = 0

        while computed_grads < num_grads:
            for sample, target in self._data_loader:
                # run sample forward and backwards pass
                model_outputs = self.module_forward(module, sample)
                self.module_backward(model_outputs, target)

                # yield so gradients can be collected
                computed_grads += 1
                yield computed_grads

                if computed_grads >= num_grads:
                    break
                module.zero_grad()


@dataclass
class MFACOptions:
    """
    Options for running the Matrix-Free Approxmiate Curvature (M-FAC) algorithm

    :param num_grads: number of gradients to store in buffer for Fisher computation.
        can be an int where that constant value will be used throughout pruning or a
        dictionary of float sparsity values to the number of gradients that should be
        stored when that sparsity level (between 0.0 and 1.0) is reached. If a
        dictionary, then 0.0 must be included as a key for the base number of gradients
        to store (i.e. {0: 64, 0.5: 128, 0.75: 256}). Default is 64
    :param damp: dampening factor, default is 1e-5
    :param grads_device: device to store the gradient buffer on. Default is "cpu"
    :param fisher_block_size: optional value to enable blocked computation of the
        Fisher matrix. Blocks will be formed consecutively along the diagonal. If
        None, blocked computation is not used. Default is 2000
    :param num_pages: number of pages to break the gradient samples into for GPU
        computation. Only available when blocked computation is not enabled.
        Default is 1
    :param available_gpus: list of GPU device names to perform computation on. Default
        is empty
    """

    num_grads: Union[Dict[float, int], int] = 64
    damp: float = 1e-5
    grads_device: Union[str, int] = "cpu"
    fisher_block_size: Optional[int] = 2000
    num_pages: int = 1  # break computation into pages when block size is None
    available_gpus: List[str] = field(default_factory=list)

    def get_num_grads_for_sparsity(self, sparsity: Union[float, List[float]]) -> int:
        if isinstance(self.num_grads, int):
            return self.num_grads
        if isinstance(sparsity, List):
            sparsity = sum(sparsity) / len(sparsity)

        sparsity_thresholds = list(sorted(self.num_grads, key=lambda key: float(key)))
        if 0.0 not in sparsity_thresholds:
            raise ValueError(
                "Dictionary of sparsity thresholds to number of grads given for "
                "MFACOptions.num_grads, but 0 not included as a sparsity threshold. "
                "0.0 must be included as a sparsity threshold. Given thresholds "
                f"{sparsity_thresholds}"
            )

        idx = 0
        while (
            idx < len(sparsity_thresholds)
            and float(sparsity_thresholds[idx]) < sparsity
        ):
            idx += 1
        idx = min(idx, len(self.num_grads) - 1)
        return self.num_grads[sparsity_thresholds[idx]]


class FisherInverse(ABC):
    """
    Abstract class for working with the inverse Fisher information matrix. Storing
    the full matrix is not a requirement.
    """

    @abstractmethod
    def diag(self) -> Tensor:
        """
        :return: the entries along the diagonal entries of the inverse Fisher matrix
        """
        raise NotImplementedError()

    @abstractmethod
    def mul(self, x: Tensor) -> Tensor:
        """
        :param x: tensor to multiply with the inverse Fisher matrix
        :return: the matrix multiplied value of x and the inverse Fisher matrix
        """
        raise NotImplementedError()


class FisherInverseFast(FisherInverse):
    """
    Base implementation of computing the inverse Fisher matrix values based on the
    M-FAC paper. Takes O(d * m) memory and O(d * m^2) time to initialize where d
    is the number of parameters and m is the number of gradient samples

    :param grads: tensor of gradient samples to compute the inverse Fisher product
        with. Dimension should be (num_samples, num_parameters)
    :param damp: the dampening factor. Default is 1e-5
    """

    def __init__(self, grads, damp=1e-5):
        self._device = grads.device
        self._dtype = grads.dtype
        self._num_samples, self._num_params = grads.shape
        self._damp = 1.0 / damp

        self._hinv_g = grads  # placeholder for grads^T * H^-1 * grads
        self._denom = torch.zeros(
            self._num_samples, device=self._device, dtype=self._dtype
        )

        grad_sample = grads[0, :].clone()
        self._hinv_g[0, :] = self._damp * grad_sample
        self._denom[0] = self._num_samples + grad_sample.dot(self._hinv_g[0, :])

        for idx in range(1, self._num_samples):
            grad_sample = grads[idx, :].clone()
            self._hinv_g[idx, :] = self._damp * grad_sample
            mul = self._hinv_g[:idx, :].matmul(grad_sample) / self._denom[:idx]
            self._hinv_g[idx, :] -= mul.matmul(self._hinv_g[:idx, :])
            self._denom[idx] = self._num_samples + grad_sample.dot(self._hinv_g[idx, :])

    def diag(self):
        """
        :return: the entries along the diagonal entries of the inverse Fisher matrix.
        """
        res = self._damp * torch.ones(
            self._num_params, device=self._device, dtype=self._dtype
        )
        for i in range(self._num_samples):
            res -= (self._hinv_g[i, :] ** 2) / self._denom[i]
        return res

    def mul(self, x):
        """
        :param x: tensor to multiply with the inverse Fisher matrix
        :return: the matrix multiplied value of x and the inverse Fisher matrix
        """
        res = self._damp * x
        mul = self._hinv_g.matmul(x) / self._denom
        res -= mul.matmul(self._hinv_g)
        return res

    def to(self, device):
        """
        :param device: device to move intermediate results to
        :return: device movement done in place, returns a copy of this object as well
        """
        # in-place
        self._hinv_g = self._hinv_g.to(device)
        self._denom = self._denom.to(device)
        self._device = device
        return self


class FisherInverseFastBlock(FisherInverse):
    """
    Implementation of computing the inverse Fisher matrix values based on the
    M-FAC paper using a given block size to break up computation. Individual
    blocks must fit into GPU memory.

    :param grads: tensor of gradient samples to compute the inverse Fisher product
        with. Dimension should be (num_samples, num_parameters)
    :param block_size: size of blocks to form along diagonal of the Fisher matrix
    :param damp: the dampening factor. Default is 1e-5
    :param devices: list of GPU device ids to use for computation. Default is to use cpu
    """

    def __init__(self, grads, block_size, damp=1e-5, devices=None):
        self._dtype = grads.dtype
        self._block_size = block_size
        self._devices = devices or ["cpu"]

        self._fisher_inv_blocks = []
        lock = threading.Lock()

        # run block computations in parallel across devices
        threads = [None] * len(self._devices)
        thread_fisher_inv_blocks = [None] * len(self._devices)

        def _compute_fisher_inv_block(block_start_idx, thread_idx_):
            block = (
                grads[:, block_start_idx : (block_start_idx + self._block_size)]
                .to(self._devices[thread_idx_])
                .contiguous()
            )
            fisher_inv_block = FisherInverseFast(block, damp=damp)
            with lock:
                # ignoring flake8 warning since thread_fisher_inv_blocks is safely
                # deleted after all calls to _compute_fisher_inv_block
                thread_fisher_inv_blocks[thread_idx_] = fisher_inv_block  # noqa: F821

        for idx, off in enumerate(range(0, grads.shape[1], self._block_size)):
            # create thread
            thread_idx = idx % len(self._devices)
            threads[thread_idx] = threading.Thread(
                target=_compute_fisher_inv_block,
                args=(off, thread_idx),
            )
            # run all threads on last iteration or when devices will be full
            if (
                thread_idx == len(self._devices) - 1
                or off + self._block_size >= grads.shape[1]
            ):
                for t in threads[: (thread_idx + 1)]:
                    t.start()
                for t in threads[: (thread_idx + 1)]:
                    t.join()
                self._fisher_inv_blocks.extend(
                    [
                        fisher_inv_block.to("cpu")
                        for fisher_inv_block in thread_fisher_inv_blocks[
                            : (thread_idx + 1)
                        ]
                        if fisher_inv_block is not None
                    ]
                )
                torch.cuda.empty_cache()

        # free h_inv_blocks from GPU memory
        del thread_fisher_inv_blocks
        torch.cuda.empty_cache()

    def diag(self):
        """
        :return: the entries along the diagonal entries of the inverse Fisher matrix.
        """
        res = []
        for idx, fisher_inv_block in enumerate(self._fisher_inv_blocks):
            device = self._devices[idx % len(self._devices)]
            fisher_inv_block = fisher_inv_block.to(device)
            res.append(fisher_inv_block.diag().to("cpu"))
            res.append(torch.zeros(0, dtype=self._dtype, device="cpu"))
            # free GPU mem
            fisher_inv_block.to("cpu")
            torch.cuda.empty_cache()
        return torch.cat(res[:-1])

    def mul(self, x):
        """
        :param x: tensor to multiply with the inverse Fisher matrix
        :return: the matrix multiplied value of x and the inverse Fisher matrix
        """
        x = x.to("cpu")
        res = []
        for idx, fisher_inv_block in enumerate(self._fisher_inv_blocks):
            device = self._devices[idx % len(self._devices)]
            fisher_inv_block = fisher_inv_block.to(device)
            x_block = x[(self._block_size * idx) : (self._block_size * (idx + 1))].to(
                device
            )
            res.append(fisher_inv_block.mul(x_block).to("cpu"))

            # free GPU mem
            fisher_inv_block.to("cpu")
            torch.cuda.empty_cache()
        return torch.cat(res)


class FisherInverseFastPageSwap(FisherInverse):
    """
    Implementation of computing the inverse Fisher matrix values based on the
    M-FAC paper using a given page size to break up computation across samples.
    Pages of gradients must fit into GPU memory.

    :param grads: tensor of gradient samples to compute the inverse Fisher product
        with. Dimension should be (num_samples, num_parameters)
    :param damp: the dampening factor. Default is 1e-5
    :param num_pages: number of pages to break gradient samples into. the number of
        gradients must be divisible by num_pages
    :param devices: list of GPU device ids to use for computation. Default is to use cpu
    """

    def __init__(self, grads, damp=1e-5, num_pages=1, devices=None):
        assert torch.cuda.is_available(), (
            "CUDA enabled device not available, "
            "but is required for using FisherInverseFastPageSwap"
        )
        self._devices = devices or ["cuda:0"]
        self._gpu0 = self._devices[0]  # for computations that fit on single GPU

        self._dtype = grads.dtype
        self._num_samples, self._num_params = grads.shape
        self._damp = 1.0 / damp
        if self._num_samples < num_pages:
            raise ValueError("num_grads cannot be smaller than num_pages")
        if self._num_samples % num_pages != 0:
            raise ValueError(
                f"num_grads {self._num_samples} must be divisible by "
                f"num_pages {num_pages}"
            )
        self._samples_per_page = self._num_samples // num_pages
        self._params_per_device = int(math.ceil(self._num_params / len(self._devices)))

        self._hinv_g = grads
        self._denom = torch.zeros(self._num_samples, dtype=self._dtype, device="cpu")

        # compute fisher inverse for first page across all GPUs
        self._comp_first_page()

        # run updates to fisher inverse on main GPU for remaining pages
        self._fisher_update_buffer = torch.zeros(
            (self._samples_per_page, self._num_params), dtype=self._dtype, device="cpu"
        )
        for page_offset in range(
            self._samples_per_page, self._num_samples, self._samples_per_page
        ):
            self._comp_page(page_offset)
        del self._fisher_update_buffer
        torch.cuda.empty_cache()

        self._denom = self._denom.to(self._gpu0)

    def diag(self):
        """
        :return: the entries along the diagonal entries of the inverse Fisher matrix.
        """
        res = self._damp * torch.ones(
            self._num_params, device=self._gpu0, dtype=self._dtype
        )
        for page_offset in range(0, self._num_samples, self._samples_per_page):
            hinv_g_page = self._hinv_g[
                page_offset : (self._samples_per_page + page_offset), :
            ].to(self._gpu0)
            for page_sample_idx in range(self._samples_per_page):
                res -= (hinv_g_page[page_sample_idx, :] ** 2) / self._denom[
                    page_sample_idx + page_offset
                ]
            del hinv_g_page

        torch.cuda.empty_cache()
        return res

    def mul(self, x):
        """
        :param x: tensor to multiply with the inverse Fisher matrix
        :return: the matrix multiplied value of x and the inverse Fisher matrix
        """
        x = x.to(self._gpu0)
        res = self._damp * x
        for page_offset in range(0, self._num_samples, self._samples_per_page):
            hinv_g_page = self._hinv_g[
                page_offset : (self._samples_per_page + page_offset), :
            ].to(self._gpu0)
            mul = (
                hinv_g_page.matmul(x)
                / self._denom[page_offset : (self._samples_per_page + page_offset)]
            )
            res -= mul.matmul(hinv_g_page)
            del hinv_g_page
        torch.cuda.empty_cache()
        return res

    def _comp_first_page(self):
        # move first page value to devices across GPUs
        def _get_first_page_on_device(params_idx, device):
            return self._hinv_g[
                : self._samples_per_page,
                params_idx : (params_idx + self._params_per_device),
            ].to(device)

        first_page_hinv_g_dist = parallel_apply(
            [_get_first_page_on_device] * len(self._devices),
            list(
                zip(range(0, self._num_params, self._params_per_device), self._devices)
            ),
        )

        # compute value for first gradient sample
        def _process_first_sample(first_page_hinv_g):
            first_grad = first_page_hinv_g[0, :].clone()
            first_page_hinv_g[0, :] = self._damp * first_grad
            self._denom[0] += first_grad.dot(first_page_hinv_g[0, :]).to("cpu")

        parallel_apply(
            [_process_first_sample] * len(self._devices),
            first_page_hinv_g_dist,
        )
        self._denom[0] += self._num_samples

        for sample_idx in range(1, self._samples_per_page):
            # update the other page gradients in parallel with two steps
            self._mul_tmp = torch.zeros(sample_idx, device="cpu", dtype=self._dtype)
            self._sample_grads_dist = [None] * len(self._devices)  # type: List[Tensor]

            def _calc_mul_update_dist(device_idx, hinv_g_shard):
                self._sample_grads_dist[device_idx] = hinv_g_shard[
                    sample_idx, :
                ].clone()
                hinv_g_shard[sample_idx, :] = (
                    self._damp * self._sample_grads_dist[device_idx]
                )
                self._mul_tmp += (
                    hinv_g_shard[:sample_idx, :]
                    .matmul(self._sample_grads_dist[device_idx])
                    .to("cpu")
                )

            parallel_apply(
                [_calc_mul_update_dist] * len(self._devices),
                list(enumerate(first_page_hinv_g_dist)),
            )
            self._mul_tmp /= self._denom[:sample_idx]

            def _apply_mul_update_dist(device_idx, hinv_g_shard):
                hinv_g_shard[sample_idx, :] -= self._mul_tmp.to(
                    hinv_g_shard.device
                ).matmul(hinv_g_shard[:sample_idx, :])
                self._denom[sample_idx] += (
                    self._sample_grads_dist[device_idx]
                    .dot(hinv_g_shard[sample_idx, :])
                    .to("cpu")
                )

            parallel_apply(
                [_apply_mul_update_dist] * len(self._devices),
                list(enumerate(first_page_hinv_g_dist)),
            )
            self._denom[sample_idx] += self._num_samples
        del self._mul_tmp
        del self._sample_grads_dist

        def _update_main_hinv_g(shard_param_idx, hinv_g_shard):
            self._hinv_g[
                : self._samples_per_page,
                shard_param_idx : (shard_param_idx + self._params_per_device),
            ] = hinv_g_shard.to("cpu")

        parallel_apply(
            [_update_main_hinv_g] * len(first_page_hinv_g_dist),
            list(
                zip(
                    range(0, self._num_params, self._params_per_device),
                    first_page_hinv_g_dist,
                ),
            ),
        )
        del first_page_hinv_g_dist

    def _comp_page(self, page_offset):
        # update fisher update buffer
        for prev_page_offset in range(0, page_offset, self._samples_per_page):
            prev_page_hinv_g = self._hinv_g[
                prev_page_offset : (self._samples_per_page + prev_page_offset), :
            ].to(self._gpu0)

            for page_sample_idx in range(self._samples_per_page):
                grad_sample = self._hinv_g[page_sample_idx + page_offset, :].to(
                    self._gpu0
                )
                mul = prev_page_hinv_g.matmul(grad_sample) / self._denom[
                    prev_page_offset : (self._samples_per_page + prev_page_offset)
                ].to(self._gpu0)
                mul = mul.matmul(prev_page_hinv_g)
                if prev_page_offset == 0:
                    self._fisher_update_buffer[page_sample_idx, :] = (
                        self._damp * grad_sample - mul
                    ).to("cpu")
                else:
                    self._fisher_update_buffer[page_sample_idx, :] -= mul.to("cpu")
            del prev_page_hinv_g

        # move buffer to main GPU and update the fisher inv state
        fisher_inv_buf_gpu = self._fisher_update_buffer.to(self._gpu0)

        grad_sample = self._hinv_g[page_offset, :].to(self._gpu0)
        self._denom[page_offset] = self._num_samples + grad_sample.dot(
            fisher_inv_buf_gpu[0, :]
        )

        for page_sample_idx in range(1, self._samples_per_page):
            grad_sample = self._hinv_g[page_sample_idx + page_offset, :].to(self._gpu0)
            mul = fisher_inv_buf_gpu[:page_sample_idx, :].matmul(
                grad_sample
            ) / self._denom[page_offset : (page_sample_idx + page_offset)].to(
                self._gpu0
            )
            fisher_inv_buf_gpu[page_sample_idx, :] -= mul.matmul(
                fisher_inv_buf_gpu[:page_sample_idx, :]
            )
            self._denom[
                page_sample_idx + page_offset
            ] = self._num_samples + grad_sample.dot(
                fisher_inv_buf_gpu[page_sample_idx, :]
            )

        # update main tensor
        self._hinv_g[
            page_offset : (self._samples_per_page + page_offset), :
        ] = fisher_inv_buf_gpu.to("cpu")
        del fisher_inv_buf_gpu


def compute_hessian_inv(
    grads: Tensor,
    mfac_options: Optional[MFACOptions] = None,
) -> FisherInverse:
    """
    :param grads: tensor of gradient samples to compute the Hessian inverse
        representation with. Should have shape (num_samples, num_parameters)
    :param mfac_options: MFACOptions object specifying how to perform the computations
    :return: FisherInverse object with access to the diagonal multiplication of the
        Fisher approximation of the Hessian inverse
    """
    if not mfac_options:
        mfac_options = MFACOptions()
    if mfac_options.fisher_block_size:
        return FisherInverseFastBlock(
            grads,
            mfac_options.fisher_block_size,
            damp=mfac_options.damp,
            devices=mfac_options.available_gpus,
        )
    elif mfac_options.available_gpus or mfac_options.num_pages > 1:
        return FisherInverseFastPageSwap(
            grads,
            damp=mfac_options.damp,
            num_pages=mfac_options.num_pages,
            devices=mfac_options.available_gpus,
        )
    else:
        return FisherInverseFast(grads, damp=mfac_options.damp)
