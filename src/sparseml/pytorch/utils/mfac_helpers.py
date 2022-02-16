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

import logging
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
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

import GPUtil


_LOGGER = logging.getLogger(__name__)
BYTES_IN_MIB = 1024 ** 2

__all__ = [
    "GradSampler",
    "MFACOptions",
    "FisherInverse",
    "FisherInverseFast",
    "FisherInverseFastBlock",
    "FisherInverseFastPageSwap",
    "FisherInverseFastSmallBlocks",
    "compute_hessian_inv",
]


class GradSampler:
    """
    Class for computing gradient samples for a Model given a sample data loader and
    loss function.

    :param data_loader: iterator of data samples to use as model inputs and their loss
        targets. items must be tuples of
        (forward_args: List, forward_kwargs: Dict, loss_targets: Any)
        where the forward pass will be outputs = model(*forward_args, **forward_kwargs)
        and loss will be loss = loss_fn(outputs, loss_targets)
    :param loss_fn: function to be called on model outputs to compute the loss at
        each step
    """

    def __init__(
        self,
        data_loader: Iterator[Tuple[List[Any], Dict[str, Any], Any]],
        loss_fn: Callable[[Any], Any],
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
            for forward_args, forward_kwargs, loss_target in self._data_loader:
                module.zero_grad()
                # run sample forward and backwards pass
                model_outputs = module(*forward_args, **forward_kwargs)
                loss = self._loss_fn(model_outputs, loss_target)
                loss.backward()

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

        _LOGGER.debug("Starting FisherInverseFastBlock")
        for block_start_idx in range(0, grads.shape[1], self._block_size):
            block = (
                grads[:, block_start_idx : (block_start_idx + self._block_size)]
                .to(self._devices[0])
                .contiguous()
            )

            fisher_inv_block = FisherInverseFast(block, damp=damp)
            self._fisher_inv_blocks.append(fisher_inv_block.to("cpu"))
            del block
        _LOGGER.debug("FisherInverseFastBlock H^-1 Calculation Complete")

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


class FisherInverseFastSmallBlocks(FisherInverse):
    """
    Implementation of computing the inverse Fisher matrix values based on the
    M-FAC paper that is optimized for speed for small block sizes

    :param grads: tensor of gradient samples to compute the inverse Fisher product
        with. Dimension should be (num_samples, num_parameters)
    :param block_size: size of blocks to form along diagonal of the Fisher matrix
    :param damp: the dampening factor. Default is 1e-5
    :param devices: list of GPU device ids to use for computation. Default is to use cpu
    :param alpha: alpha value for add step
    """

    def __init__(
        self,
        grads: Tensor,
        block_size: int,
        damp: float = 1e-5,
        devices: List[torch.device] = None,
        alpha: float = 0.0,
    ):
        self._dtype = grads.dtype
        self._element_size = grads.element_size()
        self._block_size = block_size
        self._devices = devices or ["cpu"]
        self._alpha = alpha
        self._damp = damp

        self._num_samples, self._num_params = grads.shape
        self._num_blocks = math.ceil(self._num_params / block_size)
        self._num_devices = len(self._devices)
        self._hinvs = []
        block_mem = _block_memory_size(self._block_size, self._element_size)

        cpu = self._devices[0] == "cpu"
        self.hinv(tensor=grads, block_mem=block_mem, cpu=cpu)

    def block_wise_decorator(func):
        @wraps(func)
        def wrapper_blocked(
            self,
            tensor: Tensor,
            block_mem: int,
            safety_margin: float = 0.1,
            cpu: bool = False,
        ):
            """
            Wraps the most memory intensive Fisher computations in a memory-aware block
            allocation function. The decorator will allocate a number of blocks which
            will maximize GPU memory utilization (if GPUs are utilized) with a safety
            margin

            Note: currently each device is called in sequence. There is no clear benefit
            to this regime over simply re-using one device, but it may lend to easier
            parallelization in the future and it upholds the M-FAC "available_devices"
            parameter expected behavior.

            :param tensor: The input tensor for func, the fisher computation function
            :param block_mem: The amount of memory needed (in bytes) for the
            computation of one block
            :param safety_margin: The total number of blocks allocated per device is
            (1 - safety_margin)*max_blocks, where max_blocks is the maximum that could
            fit on the device at this time
            :param cpu: When true all computation is done on the CPU, without the
            memory-aware logic
            """
            if cpu:
                self._num_blocks_per_device_call = [self._num_blocks]
                func(self, tensor, 0, "cpu")  # Process all the blocks in one call
            else:
                self._num_blocks_per_device_call = []
                self._remaining_blocks = self._num_blocks
                self._device_suite_calls = 0  # Number of calls to the full set of gpus
                # Calculate free memory available on each device
                free_device_memory = _get_free_gpu_memory(
                    _cuda_list_to_idx(self._devices)
                )
                while self._remaining_blocks > 0:
                    # Allocate blocks based on device memory, until either all blocks
                    # are allocated or all gpus have been assigned for this iteration
                    for idx, device in enumerate(self._devices):
                        self._num_blocks_per_device_call.append(
                            min(
                                self._remaining_blocks,
                                math.floor(
                                    (1 - safety_margin)
                                    * free_device_memory[idx]
                                    * BYTES_IN_MIB
                                    / block_mem
                                ),
                            )
                        )
                        self._remaining_blocks -= self._num_blocks_per_device_call[-1]
                        _LOGGER.debug(
                            f"Allocating {self._num_blocks_per_device_call[-1]} blocks to"
                            f"device {device}. {self._remaining_blocks} blocks remaining"
                        )
                        if self._remaining_blocks <= 0:
                            break

                    # Iterate through each device and perform computation
                    for idx, device in enumerate(self._devices):
                        call_idx = idx + self._device_suite_calls * self._num_devices
                        if call_idx >= len(self._num_blocks_per_device_call):
                            break
                        func(self, tensor, call_idx, device)

                    self._device_suite_calls += 1

                    # At the end of each iteration the net free memory change should be 0
                    # If the free memory decreases, throw a warning in debug mode
                    prev_free_memory = free_device_memory
                    free_device_memory = _get_free_gpu_memory(
                        _cuda_list_to_idx(self._devices)
                    )
                    for i in range(len(free_device_memory)):
                        if free_device_memory[i] < prev_free_memory[i]:
                            _LOGGER.debug(
                                f"WARNING - GPU memory not cleanly freed."
                                f"Found {(prev_free_memory[i] - free_device_memory[i])/BYTES_IN_MIB} less MiB"
                                f"since the last iteration"
                            )

                if sum(self._num_blocks_per_device_call) != self._num_blocks:
                    _LOGGER.debug(
                        "WARNING - Number of blocks processed does not equal to total number of"
                        "blocks."
                        f"Total blocks - {self._num_blocks}"
                        f"Processed blocks - {sum(self._num_blocks_per_device_call)}"
                    )

        return wrapper_blocked

    @block_wise_decorator
    def hinv(self, grads: Tensor, call_idx: int, device: str):
        """
        Initialize the H^-1 and compute its result for the given device.

        :param grads: The sampled gradients used for H^-1 computation
        :param call_idx: The index of the number of single-device calls
        :param device: the device on which to perform the computations
        """
        # initialize H_invs on each device
        num_blocks = self._num_blocks_per_device_call[call_idx]
        try:
            self._hinvs.append(
                self._init_hinv(num_blocks, self._damp, device, self._dtype)
            )
            _LOGGER.debug(f"Initialized H^-1 for {num_blocks} blocks on {device}")
        # As a failsafe for a memory issue, try again with half the number of blocks
        # This condition has not been encountered in testing as of yet
        except Exception as error_msg:
            _LOGGER.debug(
                f"{error_msg}"
                f"Initialization of H^-1 for {num_blocks} blocks on {device} failed"
                f"Retrying with {num_blocks//2} blocks"
            )
            self._hinvs.append(
                self._init_hinv(num_blocks // 2, self._damp, device, self._dtype)
            )
            self._num_blocks_per_device_call[call_idx] //= 2
            self._remaining_blocks += self._num_blocks_per_device_call[call_idx]
            _LOGGER.debug(
                f"Initialized H^-1 for {num_blocks//2} blocks on {device}"
                f"remaining blocks increased to {self._remaining_blocks}"
            )

        # build hinv_g values from grad samples
        _LOGGER.debug(
            "Calculating H^-1 with {self._num_samples} samples for call {call_idx}"
        )
        for sample_idx in range(self._num_samples):
            self._add(grads[sample_idx, :], device, call_idx)
        self._hinvs[call_idx] = self._hinvs[call_idx].to("cpu")
        _LOGGER.debug("Finished H^-1 calculation and moved mat to CPU")

        return None

    def diag(self) -> Tensor:
        """
        :return: the entries along the diagonal entries of the inverse Fisher matrix
        """
        diag_slices = [
            torch.diagonal(self._hinvs[idx], dim1=1, dim2=2).reshape(
                -1
            )  # move all to same device after computation
            for idx in range(len(self._num_blocks_per_device_call))
        ]
        return torch.cat(diag_slices)[: self._num_params]

    def mul(self, x: Tensor) -> Tensor:
        """
        :param x: tensor to multiply with the inverse Fisher matrix
        :return: the matrix multiplied value of x and the inverse Fisher matrix
        """
        x = self._pad(x).reshape((-1, self._block_size)).unsqueeze(2)
        self._mul_slices = []
        block_mem = _block_memory_size(self._block_size, self._element_size)
        cpu = self._devices[0] == "cpu"
        self.mul_blocked(tensor=x, block_mem=block_mem, cpu=cpu)
        return torch.cat(self._mul_slices)[: self._num_params]

    @block_wise_decorator
    def mul_blocked(self, x: Tensor, call_idx: int, device: str) -> Tensor:
        """
        :param x: tensor to multiply with the inverse Fisher matrix
        :param call_idx: The index of the number of single-device calls
        :param device: the device on which to perform the computations
        :return: the matrix multiplied value of x and the inverse Fisher matrix
        """
        x_slice = x[
            int(
                torch.sum(
                    torch.tensor(self._num_blocks_per_device_call[:call_idx])
                ).item()
            ) : int(
                torch.sum(
                    torch.tensor(self._num_blocks_per_device_call[: call_idx + 1])
                ).item()
            )
        ].to(device)

        # Get the H^-1 values corresponding to the number of blocks used here.
        # It's clunky compared to torch.cat()[idx], but avoids duplicating
        # the memory of H^-1
        start_block = sum(self._num_blocks_per_device_call[:call_idx])
        end_block = sum(self._num_blocks_per_device_call[: call_idx + 1])
        t_hinv = []
        tensor_start = 0
        tensor_end = 0
        for tensor in self._hinvs:
            tensor_end += len(tensor)
            if start_block > tensor_end:
                continue
            if end_block < tensor_end:
                t_hinv.append(
                    tensor[start_block - tensor_start : end_block - tensor_start]
                )
                break
            else:
                t_hinv.append(tensor[start_block - tensor_start :])
                start_block = tensor_end
            tensor_start = tensor_end

        mul_slice = (
            torch.bmm(torch.cat(t_hinv).to(device), x_slice)
            .reshape(-1)
            .to("cpu")  # move all to same device after computation
        )
        self._mul_slices.append(mul_slice)

    def _init_hinv(
        self,
        num_blocks: int,
        damp: float,
        device: torch.device,
        dtype: torch.dtype,
    ):
        # initialize hinv to num_blocks diagonal blocks of size blocksize
        base_block = torch.diag(
            torch.full([self._block_size], 1.0 / damp, dtype=dtype, device=device)
        )
        return torch.repeat_interleave(base_block.unsqueeze(0), num_blocks, 0)

    def _add(self, grad_sample: Tensor, device, call_idx):
        # add gradient sample into H_invs
        num_params_per_device = [
            num_blocks_device * self._block_size
            for num_blocks_device in self._num_blocks_per_device_call
        ]

        grad_sample_slice = grad_sample[
            int(torch.sum(torch.tensor(num_params_per_device[:call_idx])).item()) : int(
                torch.sum(torch.tensor(num_params_per_device[: call_idx + 1])).item()
            )
        ]
        if len(grad_sample_slice) % self._block_size != 0:
            # pad to block size
            pad_vals = torch.zeros(
                self._block_size - len(grad_sample_slice) % self._block_size
            )
            grad_sample_slice = torch.cat(
                [grad_sample_slice, pad_vals.to(grad_sample.device)]
            )
        grads_blocked_device = grad_sample_slice.to(device).reshape(
            (-1, self._block_size)
        )

        hinv_g_slice = torch.bmm(
            self._hinvs[call_idx], grads_blocked_device.unsqueeze(2)
        )

        denom = (
            self._num_samples
            + torch.bmm(grads_blocked_device.unsqueeze(1), hinv_g_slice)
        ).squeeze(2)

        hinv_g_slice = hinv_g_slice.reshape(-1, self._block_size)

        for idx_block in range(self._block_size):
            # update h_inv calculation across block dims
            self._hinvs[call_idx][:, idx_block, :] -= hinv_g_slice * (
                hinv_g_slice[:, idx_block].unsqueeze(1) / denom
            )

    def _pad(self, x: Tensor):
        # pad 1-d tensor to num_blocks * block_size
        padded_x = torch.zeros(
            self._num_blocks * self._block_size,
            dtype=self._hinvs[0].dtype,
            device=self._hinvs[0].device,
        )
        padded_x[: x.size(0)] = x
        return padded_x


def compute_hessian_inv(
    grads: Tensor,
    mfac_options: Optional[MFACOptions] = None,
) -> FisherInverse:
    """
    Determine which FisherInverse algorithm to use.

    :param grads: tensor of gradient samples to compute the Hessian inverse
        representation with. Should have shape (num_samples, num_parameters)
    :param mfac_options: MFACOptions object specifying how to perform the computations
    :return: FisherInverse object with access to the diagonal multiplication of the
        Fisher approximation of the Hessian inverse
    """
    if not mfac_options:
        _LOGGER.info("No M-FAC options found - using defaults")
        mfac_options = MFACOptions()
    # The amount of memory required for the computation of one block is the main 
    # decider in the FisherInverse algorithm to use
    if mfac_options.fisher_block_size:
        block_mem_size = _block_memory_size(
            block_size=mfac_options.fisher_block_size, element_size=grads.element_size()
        )

        _LOGGER.debug(
            f"Calculated Fisher block with size {mfac_options.fisher_block_size}"
            f"to occupy {block_mem_size} bytes/ {block_mem_size/BYTES_IN_MIB} MiB in memory"
        )

        free_device_mem = _get_free_gpu_memory(
            _cuda_list_to_idx(mfac_options.available_gpus)
        )

        _LOGGER.debug(
            "Free memory on devices:"
            + "\n".join(
                [
                    f"{mfac_options.available_gpus[i]}: {str(free_device_mem[i]/BYTES_IN_MIB)}"
                    for i in range(len(free_device_mem))
                ]
            )
        )

        # Determine which of the available gpus have enough free memory to host
        # the block computation
        available_gpus = [
            gpu
            for i, gpu in enumerate(mfac_options.available_gpus)
            if free_device_mem[i] > block_mem_size / BYTES_IN_MIB
        ]

        # FisherInverseFastBlock works only in sequential mode. Unless only one block
        # or less can fit on the GPU, FisherInverseFastSmallBlocks should be used
        if len(available_gpus) > 0 or not free_device_mem:
            _LOGGER.info("Using Small Block Fast Fisher Inverse Implementation")
            _LOGGER.debug(
                "Using the following devices for M-FAC:" + "\n".join(available_gpus)
            )
            mfac_options.available_gpus = available_gpus
            block_fisher_class = FisherInverseFastSmallBlocks
        else:
            _LOGGER.info(
                "Large block size detected - Using Fast Block Fisher Inverse Implementation"
            )
            block_fisher_class = FisherInverseFastBlock

        return block_fisher_class(
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


def _get_free_gpu_memory(
    device_idx: List[int] = [], clear_cache: bool = True
) -> List[float]:
    """
    Get free memory available on device(s)

    Note: GPUtil and PyTorch may see different devices and device orders depending on
    the value of CUDA_VISIBLE_DEVICES. This function honors the PyTorch device view.

    :param device_idx: Devices to retrieve free memory for. If empty, will use
    all visible devices
    :param clear_cache: Whether to clear pytorch reserved memory before retrieving free
    memory. Leaving this flag on will result in a larger (and more accurate) free memory
    reading, but comes at a (small) cost to pytorch tensor allocation speed. In the case
    of very high frequency calls, it may be better to turn clear_cache off.
    """

    if not device_idx:
        device_idx = list(range(torch.cuda.device_count()))
    if not device_idx:
        return []  # An empty list signals to use cpu
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        if not os.environ["CUDA_VISIBLE_DEVICES"]:
            raise ValueError(
                "GPU device specified for M-FAC, but no GPUs"
                "were found in CUDA_VISIBLE_DEVICES"
            )
        gpu_idx_all = [
            int(idx) for idx in os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        ]
        gpu_idx = [gpu_idx_all[idx] for idx in device_idx]

    else:
        gpu_idx = device_idx

    if clear_cache:
        torch.cuda.empty_cache()
    gpus_all = GPUtil.getGPUs()
    return [gpus_all[idx].memoryFree for idx in gpu_idx]


def _cuda_list_to_idx(cuda_device_list: List[str]) -> List[int]:
    """
    Convert list of cuda device string names to indices.
    e.g. "cuda:0" -> 0
    """
    return [
        int("".join(filter(str.isdigit, device_str))) for device_str in cuda_device_list
    ]


def _block_memory_size(block_size: int, element_size: int) -> int:
    """
    Calculate memory needed for H^-1 calculations of one block.
    """
    # B^2 * e_size - memory required for H^-1
    # 4*B * e_size - memory required for additional comp vectors
    return (block_size ** 2 + 4 * block_size) * element_size
