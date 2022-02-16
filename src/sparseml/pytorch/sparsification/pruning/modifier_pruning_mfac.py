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
Modifier classes implementing M-FAC pruning as described in
https://arxiv.org/pdf/2107.03356.pdf
"""
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.nn import Module, Parameter
import torch.distributed as dist

from sparseml.pytorch.utils.logger import BaseLogger
from sparseml.pytorch.optim.modifier import ModifierProp
from sparseml.pytorch.optim.modifier import PyTorchModifierYAML
from sparseml.pytorch.sparsification.pruning.scorer import PruningParamsGradScorer
from sparseml.pytorch.sparsification.pruning.mask_creator import (
    PruningMaskCreator,
    UnstructuredPruningMaskCreator,
)
from sparseml.pytorch.sparsification.pruning.modifier_pruning_base import (
    BaseGradualPruningModifier,
)
from sparseml.utils import ALL_TOKEN
__all__ = [
    "MFACPruningModifier"
]

_LOGGER = logging.getLogger(__name__)

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


@PyTorchModifierYAML()
class MFACPruningModifier(BaseGradualPruningModifier):
    """
    Gradually applies kernel sparsity to a given parameter or parameters from
    init_sparsity until final_sparsity is reached over a given amount of time
    and applied with an interpolated function for each step taken.

    Uses the Matrix-Free Approxmiate Curvature (M-FAC) algorithm for solving
    for optimal pruning updates by estimating the inverse Hessian matrix to the
    loss over time under the Optimal Brain Surgeon (OBS) framework.
    A link to the paper will be included here in an upcoming update.

    | Sample yaml:
    |   !MFACPruningModifier
    |       init_sparsity: 0.05
    |       final_sparsity: 0.8
    |       start_epoch: 0.0
    |       end_epoch: 10.0
    |       update_frequency: 1.0
    |       params: ["re:.*weight"]
    |       leave_enabled: True
    |       inter_func: cubic
    |       log_types: __ALL__
    |       mask_type: unstructured
    |       mfac_options:
    |           num_grads: {0.0: 64, 0.5: 128, 0.75: 256, 0.85: 512}
    |           fisher_block_size: 10000
    |           available_gpus: ["cuda:0"]

    :param init_sparsity: the initial sparsity for the param to start with at
        start_epoch
    :param final_sparsity: the final sparsity for the param to end with at end_epoch.
        Can also be a Dict of final sparsity values to a list of parameters to apply
        them to. If given a Dict, then params must be set to [] and the params to
        be pruned will be read from the final_sparsity Dict
    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param update_frequency: The number of epochs or fraction of epochs to update at
        between start and end
    :param params: A list of full parameter names or regex patterns of names to apply
        pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
        will match to all parameters. __ALL_PRUNABLE__ will match to all ConvNd
        and Linear layers' weights. If a sparsity to param mapping is defined by
        final_sparsity, then params should be set to []
    :param leave_enabled: True to continue masking the weights after end_epoch,
        False to stop masking. Should be set to False if exporting the result
        immediately after or doing some other prune
    :param inter_func: the type of interpolation function to use:
        [linear, cubic, inverse_cubic]
    :param log_types: The loggers to allow the learning rate to be logged to,
        default is __ALL__
    :param mask_type: String to define type of sparsity (options: ['unstructured',
        'channel', 'filter']), List to define block shape of a parameters in and out
        channels, or a SparsityMaskCreator object. default is 'unstructured'
    :param global_sparsity: set True to enable global pruning. if False, pruning will
        be layer-wise. Default is False
    :param use_gradient_buffering: Optional bool to use gradient buffering instead of
    grad sampling. By default, grad sampling is always used when available
    :param mfac_options: Dictionary of key words specifying arguments for the M-FAC
        pruning run. num_grads controls the number of gradient samples that are kept,
        fisher_block_size specifies the block size to break the M-FAC computation into
        (default is 2000, use None for no blocks), available_gpus specifies a list
        of device ids that can be used for computation. For a full list of options,
        see the MFACOptions dataclass documentation. Default configuration uses
        CPU for computation without blocked computation
    """

    def __init__(
        self,
        init_sparsity: float,
        final_sparsity: Union[float, Dict[float, List[str]]],
        start_epoch: float,
        end_epoch: float,
        update_frequency: float,
        params: Union[str, List[str]],
        leave_enabled: bool = True,
        inter_func: str = "cubic",
        log_types: Union[str, List[str]] = ALL_TOKEN,
        global_sparsity: bool = False,
        use_gradient_buffering: Optional[bool] = None,
        mfac_options: Dict[str, Any] = None,
    ):
        super().__init__(
            params=params,
            init_sparsity=init_sparsity,
            final_sparsity=final_sparsity,
            inter_func=inter_func,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            log_types=log_types,
            global_sparsity=global_sparsity,
            leave_enabled=leave_enabled,
        )
        self._mfac_options = mfac_options or {}
        self._grad_sampler = None
        self._use_gradient_buffering = use_gradient_buffering

    @ModifierProp(serializable=True)
    def mfac_options(self) -> Dict[str, Any]:
        """
        :return: Dictionary of key words specifying arguments for the M-FAC
            pruning run. num_grads controls the number of gradient samples,
            fisher_block_size is the block size to break the M-FAC computation into
            (default is 2000, None means no blocks), available_gpus specifies a list
            of device ids that can be used for computation. For a full list of options,
            see the MFACOptions dataclass documentation.
        """
        return self._mfac_options

    @ModifierProp(serializable=False)
    def use_gradient_buffering(self) -> Optional[bool]:
        """
        Return flag indicating force use of gradient buffering over a gradient sampler
        """
        return self._use_gradient_buffering

    def initialize(
        self,
        module: Module,
        epoch: float = 0,
        loggers: Optional[List[BaseLogger]] = None,
        **kwargs,
    ):
        """
        Grab the layers and apply if epoch in range to control pruning for.
        If `grad_sampler: GradSampler` is present in kwargs, then will add
        it to this class and use the sampler instead of live gradient buffering

        :param module: the PyTorch model/module to modify
        :param epoch: The epoch to initialize the modifier and module at.
            Defaults to 0 (start of the training process)
        :param loggers: Optional list of loggers to log the modification process to
        :param kwargs: Optional kwargs to support specific arguments
            for individual modifiers.
        """
        _LOGGER.debug("Initializing MFACPruningModifier")
        if "grad_sampler" in kwargs and self._use_gradient_buffering is not True:
            # set grad sampler, must be done before initialize in case pruning step
            # occurs on initialize epoch
            grad_sampler = kwargs["grad_sampler"]
            if not isinstance(grad_sampler, GradSampler):
                raise ValueError(
                    "grad_sampler must be an instance of the GradSampler class"
                )
            self._grad_sampler = grad_sampler
            _LOGGER.debug("Using provided GradSampler")

        elif self._use_gradient_buffering is False:
            raise RuntimeError(
                "grad_sampler must be provided when use_gradient_buffering is set"
                "to False"
            )
        else:
            _LOGGER.debug("Using gradient buffering")

        super().initialize(module, epoch, loggers, **kwargs)

        if self._grad_sampler is not None:
            # disable gradient buffering until sampler is invoked
            self._scorer.buffer_grads = False

    def _get_mask_creator(self) -> PruningMaskCreator:
        """
        :return: mask creator object to be used by this pruning algorithm
        """
        return UnstructuredPruningMaskCreator()

    def _get_scorer(self, params: List[Parameter]) -> PruningParamsGradScorer:
        """
        :param params: list of Parameters for scorer to track
        :return: param scorer object to be used by this pruning algorithm
        """
        return MFACPruningParamsScorer(params, self._mfac_options)

    def check_mask_update(
        self, module: Module, epoch: float, steps_per_epoch: int, **kwargs
    ):
        _LOGGER.debug("Running M-FAC Pruning")
        # create grads for pne-shot pruning
        if self._grad_sampler is not None:
            self.module_masks.scorer.buffer_grads = True  # enable buffering
            self._collect_grad_samples(module, self._grad_sampler)
            self._pre_step_completed = True
            self.module_masks.scorer.buffer_grads = False  # re-disable buffering

        super()._check_mask_update(module, epoch, steps_per_epoch, **kwargs)

    def _collect_grad_samples(
        self,
        module: Module,
        grad_sampler: GradSampler,
    ):
        if not isinstance(grad_sampler, GradSampler):
            raise ValueError(
                "One-shot MFAC pruning requires a GradSampler object given by the "
                f"grad_sampler kwarg. Given an object of type {type(grad_sampler)}"
            )
        num_grads = MFACOptions(**self._mfac_options).get_num_grads_for_sparsity(
            self._applied_sparsity or 0.0
        )

        _LOGGER.debug("Starting to collect {num_grads} grads with GradSampler")
        for _ in grad_sampler.iter_module_backwards(module, num_grads):
            self._module_masks.pre_optim_step_update()
        _LOGGER.debug("GradSampler grad collection complete")

class MFACPruningParamsScorer(PruningParamsGradScorer):
    """
    Scores parameters using the Matrix-Free Approximate Curvature (M-FAC)
    algorithm to solve for the optimal update in the Optimal Brain Surgeon (OBS)
    framework.  Given an estimate of the inverse Hessian matrix H^-1,
    scores are determined by W^2 / (2 * diag(H^-1)).

    Additionally, when masking, weights should also be updated by the optimal
    perturbation: -w_i * H^-1 / H_{i,i} for every newly masked weight w_i.

    :param params: list of model Parameters to track and score
    :param mfac_options: Dictionary of key words specifying arguments for the M-FAC
        pruning run. num_grads controls the number of gradient samples that are kept,
        fisher_block_size if given enables block approximations of the Fisher matrix
        (if not specified, the full matrix is used), available_gpus specifies a list
        of device ids that can be used for computation. For a full list of options,
        see the MFACOptions dataclass documentation. Default configuration uses
        CPU for computation without blocked computation
    """

    def __init__(self, params: List[Parameter], mfac_options: MFACOptions = None):
        super().__init__(params)

        # control when to do live gradient buffering, enabled by default
        self.buffer_grads = True

        self._mfac_options = mfac_options or MFACOptions()
        self._unpruned_idxs = [None] * len(self._params)  # type: List[Tensor]
        self._grad_buffer = None  # type: Tensor
        self._grads = None  # placeholder for all grads across buffers
        self._buffer_idx = 0
        self._latest_h_inv_diag = None  # type: tuple

        # scale num_grads by number of DDP processes
        if self._is_ddp:
            world_size = dist.get_world_size()
            if isinstance(self._mfac_options.num_grads, int):
                self._mfac_options.num_grads = (
                    self._mfac_options.num_grads // world_size
                )
            else:  # dict
                self._mfac_options.num_grads = {
                    k: v // world_size for k, v in self._mfac_options.num_grads.items()
                }

        self._pickle_exclude_params.extend(
            [
                "_unpruned_idxs",
                "_grad_buffer",
                "_grads",
                "_buffer_idx",
                "_latest_h_inv_diag",
            ]
        )

    def score_parameters(self) -> List[Tensor]:
        """
        :return: List of Tensors the same shapes as the given Parameters where
            each Parameter's elements are scored based on the optimal value
            given by the OBS method. For the approximated Hessian inverse matrix
            H^-1, scores will be W^2 / (2 * diag(H^-1))
        """

        if self._grad_buffer is None or torch.any(
            torch.all(self._grad_buffer == 0.0, dim=1)
        ):
            # raise Exception if grad buffer is not full
            raise RuntimeError(
                "MFAC pruning step called, but not enough gradient samples have been "
                f"collected. Expected {self._mfac_options.num_grads} samples"
            )

        if self._is_ddp:
            # move all grads to one device
            if self._is_main_proc:
                # initialize grads tensor to fit grad buffers from all processes
                num_grads = self._grad_buffer.size(0)
                self._grads = self._grad_buffer.new_zeros(
                    (
                        num_grads * dist.get_world_size(),
                        self._grad_buffer.size(1),
                    )
                )
                # have gather list reference grads to avoid doubling memory on concat
                gather_list = [
                    self._grads[proc_idx * num_grads : (proc_idx + 1) * num_grads, :]
                    for proc_idx in range(dist.get_world_size())
                ]
                dist.gather(
                    self._grad_buffer,
                    gather_list=gather_list,
                    group=self._gloo_handle,
                    dst=0,
                )
            else:
                dist.gather(self._grad_buffer, group=self._gloo_handle, dst=0)
        else:
            self._grads = self._grad_buffer

        del self._grad_buffer  # free buffer from memory, all data moved to _grads

        if self._is_main_proc:
            param_scores = self._score_parameters()

        # broadcast scores to all processes
        to_broadcast = (
            param_scores
            if self._is_main_proc
            else [None for _ in range(len(self._params))]
        )
        param_scores = self._broadcast_list_from_main(to_broadcast)

        # put scores on correct device
        for idx, param in enumerate(self._params):
            param_scores[idx] = param_scores[idx].to(param.device)

        return param_scores

    def pre_optim_step_update(self, masks: List[Tensor]):
        """
        Update the gradient buffer based on the current gradients

        :param masks: latest masks that are applied to these parameters
        """

        if not self.buffer_grads or any(param.grad is None for param in self._params):
            # only update buffer if all gradients are computed
            return

        if self._grad_buffer is None:
            self._setup_grad_buffer(masks)

        # get non-pruned grads
        non_pruned_grads = [
            param.grad.view(-1)[self._unpruned_idxs[idx]].to(self._grad_buffer.device)
            for idx, param in enumerate(self._params)
        ]

        # update buffer
        torch.cat(
            non_pruned_grads,
            out=self._grad_buffer[self._buffer_idx, :],  # write to buffer
        )
        del non_pruned_grads

        # update buffer idx
        self._buffer_idx += 1
        self._buffer_idx %= self._grad_buffer.size(0)

    @torch.no_grad()
    def mask_update(self, masks: List[Tensor], mask_diffs: List[Tensor]):
        """
        Update parameters for a new mask based on the OBS optimal perturbation:
        -w_i * H^-1 / H_{i,i} for every newly masked weight w_i

        :param masks: latest masks to be applied to these parameters
        :param mask_diffs: mask diff values returned by mask_difference for these
            masks that describe how these masks changed since the last update
        """
        # calculate optimal perturbation on main process and broadcast to all
        perturb = self._calc_params_perterb(mask_diffs) if self._is_main_proc else None
        perturb = self._broadcast_list_from_main([perturb])[0]

        # update weights by mapping to perturbation
        weights_idx = 0
        for idx, param in enumerate(self._params):
            indices = self._unpruned_idxs[idx]
            next_idx = weights_idx + indices.numel()

            with torch.no_grad():
                param.view(-1)[self._unpruned_idxs[idx]] += perturb[
                    weights_idx:next_idx
                ].to(param.device)

            weights_idx = next_idx

        self._latest_h_inv_diag = None  # clear h_inv
        self._grads = None  # clear grads
        self._setup_grad_buffer(masks)  # reset grad buffer
        torch.cuda.empty_cache()  # release GPU memory

    def _score_parameters(self) -> List[Tensor]:
        # score params using MFAC and the gathered grad buffers
        # gather non-pruned weights
        non_pruned_weights = torch.empty(self._grads.size(1)).to(self._grads.device)
        weights_idx = 0
        for idx, param in enumerate(self._params):
            indices = self._unpruned_idxs[idx]
            next_idx = weights_idx + indices.numel()
            non_pruned_weights[weights_idx:next_idx] = param.data.view(-1)[indices]
            weights_idx = next_idx

        # inverse hessian approximation
        h_inv = compute_hessian_inv(self._grads, self._mfac_options)
        diag = h_inv.diag().to(non_pruned_weights.device)

        # compute global scores for non-pruned weights
        global_scores = (non_pruned_weights ** 2) / (2.0 * diag)
        parameter_scores = []
        minimum_score = global_scores.min().item() - 1

        # map global scores to parameter weight shapes
        weights_idx = 0
        for idx, param in enumerate(self._params):
            indices = self._unpruned_idxs[idx]
            next_idx = weights_idx + indices.numel()
            param_score = (
                torch.ones_like(param.data, device="cpu").detach().requires_grad_(False)
            )
            param_score *= minimum_score  # set values to the minimal score by default

            param_score.view(-1)[self._unpruned_idxs[idx]] = global_scores[
                weights_idx:next_idx
            ].to(param_score.device)
            weights_idx = next_idx

            parameter_scores.append(param_score)

        # save h_inv and diag for weight update later
        self._latest_h_inv_diag = (h_inv, diag)
        torch.cuda.empty_cache()  # release GPU memory

        return parameter_scores

    def _calc_params_perterb(self, mask_diffs):
        # select weights that are about to be masked with 0s for unmasked weights
        weights_to_prune = torch.zeros(
            self._grads.size(1),
            device=self._grads.device,
        )
        weights_idx = 0
        for idx, mask_diff in enumerate(mask_diffs):
            indices = self._unpruned_idxs[idx]
            next_idx = weights_idx + indices.numel()
            weights_to_prune[weights_idx:next_idx] = (
                self._params[idx].data.view(-1)[indices]
                * (mask_diff.view(-1)[indices] == -1.0)  # newly pruned weights
            ).to(weights_to_prune.device)
            weights_idx = next_idx

        # calculate optimal perturbation = -w_i * H^-1 / H_{i,i}
        h_inv, diag = self._latest_h_inv_diag
        return h_inv.mul(-1.0 * weights_to_prune / diag)

    def _setup_grad_buffer(self, masks: Tensor):
        total_nonzero = 0
        for idx, mask in enumerate(masks):
            self._unpruned_idxs[idx] = mask.view(-1).nonzero(as_tuple=False).reshape(-1)
            total_nonzero += self._unpruned_idxs[idx].numel()
        # only track nonzero grads
        num_grads = self._mfac_options.get_num_grads_for_sparsity(
            self._last_applied_sparsity
        )
        self._grad_buffer = torch.zeros(
            (num_grads, total_nonzero),
            device="cpu",
        )
        self._buffer_idx = 0


'''
Classes and methods for computing H^-1
'''

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
        self._block_size = block_size
        self._devices = devices or ["cpu"]
        self._alpha = alpha

        self._num_samples, self._num_params = grads.shape
        self._num_blocks = math.ceil(self._num_params / block_size)
        self._num_devices = len(self._devices)
        self._num_blocks_per_device_call = []
        self._hinvs = []

        # Liberal estimate of memory allocated to drivers. To get accurate memory
        # reading, gputil or similar library should be used
        static_allocated_memory_frac = 0.2
        remaining_blocks = self._num_blocks
        self._device_suite_calls = 0
        prev_free_memory = None
        torch.cuda.empty_cache()
        _LOGGER.debug("Starting FisherInverseFastSmallBlocks")
        while remaining_blocks > 0:
            for device in devices:
                free_device_memory = math.floor(
                    (
                        torch.cuda.get_device_properties(device=device).total_memory
                        - torch.cuda.memory_allocated(device=device)
                    )
                    * (1 - static_allocated_memory_frac)
                )
                prev_free_memory = free_device_memory
                _LOGGER.debug(f"Free memory on {device} - {free_device_memory}")
                if free_device_memory < prev_free_memory:
                    _LOGGER.debug(
                        f"WARNING - GPU memory not cleanly freed. Found {prev_free_memory - free_device_memory}"
                        f"less bytes / {prev_free_memory - free_device_memory/BYTES_IN_GIB}"
                        f"since the last iteration"
                    )
                self._num_blocks_per_device_call.append(
                    min(
                        remaining_blocks,
                        math.ceil(
                            free_device_memory
                            / (
                                self._block_size
                                * self._block_size
                                * grads.element_size()
                            )
                        ),
                    )
                )
                remaining_blocks -= self._num_blocks_per_device_call[-1]
                _LOGGER.debug(
                    f"Allocating {self._num_blocks_per_device_call[-1]} blocks to"
                    f"device {device}. {remaining_blocks} blocks remaining"
                )
                if remaining_blocks <= 0:
                    break

            for idx, device in enumerate(devices):
                call_idx = idx + self._device_suite_calls * self._num_devices

                # initialize H_invs on each device
                num_blocks = self._num_blocks_per_device_call[call_idx]
                try:
                    self._hinvs.append(
                        self._init_hinv(
                            num_blocks, damp, self._devices[idx], grads.dtype
                        )
                    )
                    _LOGGER.debug(
                        f"Initialized H^-1 for {num_blocks} blocks on {self._devices[idx]}"
                    )
                except Exception as error_msg:
                    _LOGGER.debug(
                        f"{error_msg}"
                        f"Initialization of H^-1 for {num_blocks} blocks on {self._devices[idx]} failed"
                        f"Retrying with {num_blocks//2} blocks"
                    )
                    self._hinvs.append(
                        self._init_hinv(
                            num_blocks // 2, damp, self._devices[idx], grads.dtype
                        )
                    )
                    self._num_blocks_per_device_call[call_idx] //= 2
                    remaining_blocks += self._num_blocks_per_device_call[call_idx]
                    _LOGGER.debug(
                        f"Initialized H^-1 for {num_blocks//2} blocks on {self._devices[idx]}"
                        f"remaining blocks increased to {remaining_blocks}"
                    )

                # build hinv_g values from grad samples
                _LOGGER.debug(
                    "Calculating H^-1 with {self._num_samples} samples for call {call_idx}"
                )
                for sample_idx in range(self._num_samples):
                    self._add(grads[sample_idx, :], device, call_idx)
                self._hinvs[call_idx] = self._hinvs[call_idx].to("cpu")
                _LOGGER.debug("Finished H^-1 calculation and moved mat to CPU")

            self._device_suite_calls += 1

        if sum(self._num_blocks_per_device_call) != self._num_blocks:
            _LOGGER.debug(
                "WARNING - Number of blocks processed does not equal to total number of"
                "blocks."
                f"Total blocks - {self._num_blocks}"
                f"Processed blocks - {sum(self._num_blocks_per_device_call)}"
            )
        _LOGGER.debug("FisherInverseFastSmallBlocks call complete")

    def diag(self) -> Tensor:
        """
        :return: the entries along the diagonal entries of the inverse Fisher matrix
        """
        diag_slices = [
            torch.diagonal(self._hinvs[idx], dim1=1, dim2=2)
            .reshape(-1)
            .to(self._devices[0])  # move all to same device after computation
            for idx in range(len(self._num_blocks_per_device_call))
        ]
        return torch.cat(diag_slices)[: self._num_params]

    def mul(self, x: Tensor) -> Tensor:
        """
        :param x: tensor to multiply with the inverse Fisher matrix
        :return: the matrix multiplied value of x and the inverse Fisher matrix
        """
        x = self._pad(x).reshape((-1, self._block_size)).unsqueeze(2)
        mul_slices = []
        for idx in range(len(self._num_blocks_per_device_call)):
            device = self._devices[idx % self._num_devices]
            x_slice = x[
                int(
                    torch.sum(
                        torch.tensor(self._num_blocks_per_device_call[:idx])
                    ).item()
                ) : int(
                    torch.sum(
                        torch.tensor(self._num_blocks_per_device_call[: idx + 1])
                    ).item()
                )
            ].to(device)
            mul_slice = (
                torch.bmm(self._hinvs[idx].to(device), x_slice)
                .reshape(-1)
                .to(self._devices[0])  # move all to same device after computation
            )
            mul_slices.append(mul_slice)
        return torch.cat(mul_slices)[: self._num_params]

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
    :param grads: tensor of gradient samples to compute the Hessian inverse
        representation with. Should have shape (num_samples, num_parameters)
    :param mfac_options: MFACOptions object specifying how to perform the computations
    :return: FisherInverse object with access to the diagonal multiplication of the
        Fisher approximation of the Hessian inverse
    """
    if not mfac_options:
        _LOGGER.info("No M-FAC options found - using defaults")
        mfac_options = MFACOptions()
    if mfac_options.fisher_block_size:
        block_mem_size = (
            mfac_options.fisher_block_size ** 2 + 4 * mfac_options.fisher_block_size
        ) * grads.element_size()
        _LOGGER.debug(
            f"Calculated Fisher block with size {mfac_options.fisher_block_size}"
            f"to occupy {block_mem_size} bytes/ {block_mem_size/BYTES_IN_GIB} GiB in memory"
        )
        free_device_mem = sum(
            [
                torch.cuda.get_device_properties(device=device).total_memory
                - torch.cuda.memory_allocated(device=device)
                for device in mfac_options.available_gpus
            ]
        )
        _LOGGER.debug(
            f"Total free memory on devices: {mfac_options.available_gpus}"
            f"found to be {free_device_mem/BYTES_IN_GIB} GiB"
        )

        # FisherInverseFastBlock works only in sequential mode. Unless only one block
        # or less can fit on the GPU, FisherInverseFastSmallBlocks should be used
        if block_mem_size * 2 < free_device_mem:
            _LOGGER.info("Using Small Block Fast Fisher Inverse Implementation")
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