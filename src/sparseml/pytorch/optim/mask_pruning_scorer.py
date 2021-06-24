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
Classes for tracking and scoring model parameters to generate pruning scores
"""


from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Parameter

from sparseml.pytorch.utils import MFACOptions, compute_hessian_inv


__all__ = [
    "AVALIABLE_SCORER_CLASSES",
    "PruningParamsScorer",
    "MagnitudePruningParamsScorer",
    "MovementPruningParamsScorer",
    "MFACPruningParamsScorer",
    "MFACOptions",
    "create_pruning_param_scorer",
]


class PruningParamsScorer(ABC):
    """
    Base abstract class for scoring model parameters for pruning

    :param params: list of model Parameters to track and score
    """

    def __init__(self, params: List[Parameter]):
        self._params = params
        self._last_applied_sparsity = 0.0

    @abstractmethod
    def score_parameters(self) -> List[Tensor]:
        """
        :return: List of Tensors the same shapes as the given Parameters that
            correspond to their scores to be pruned by
        """
        raise NotImplementedError()

    def pre_optim_step_update(self, masks: List[Tensor]):
        """
        Perform any required logic for tracking Parameter data and gradients before
            an Optimizer step is applied to the model.

        :param masks: latest masks that are applied to these parameters
        """
        pass

    def mask_update(self, masks: List[Tensor], mask_diffs: List[Tensor]):
        """
        Perform any updates based on the latest mask to be applied to the weights
        immediately after this function completes

        :param masks: latest masks to be applied to these parameters
        :param mask_diffs: mask diff values returned by mask_difference for these
            masks that describe how these masks changed since the last update
        """
        pass

    def update_last_applied_sparsity(self, sparsity: float):
        """
        :param sparsity: sparsity level between 0.0 and 1.0 that was the last value
            set for the given parameters
        """
        self._last_applied_sparsity = sparsity

    def check_regen_param_vals(self):
        """
        Check that all variables based on the params are on the correct device
        and regenerate if not
        """
        pass

    def on_pruning_end(self):
        """
        Perform any cleanup after pruning is complete
        """
        pass

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """
        :return: name of this pruning method
        """
        raise NotImplementedError()


class PruningParamsGradScorer(PruningParamsScorer, ABC):
    """
    Abstract class for PruningParamsScorers that use gradients to score parameters.
    Adds extra abstraction for handling gradient sharing between parameters

    :param params: list of model Parameters to track and score
    """

    def __init__(self, params: List[Parameter]):
        super().__init__(params=params)

        self._is_ddp = dist.is_initialized()
        self._is_main_proc = not self._is_ddp or dist.get_rank() == 0

        # create group to broadcast gradients across processes
        self._gloo_handle = dist.new_group(backend="gloo") if self._is_ddp else None

        self._pickle_exclude_params = ["_is_ddp", "_is_main_proc", "_gloo_handle"]

    def __getstate__(self) -> Dict[str, Any]:
        """
        :return: state of this object as dict, without DDP related parameters
        """
        return {
            param: val
            for param, val in self.__dict__.items()
            if param not in self._pickle_exclude_params
        }

    def on_pruning_end(self):
        """
        Perform any cleanup after pruning is complete
        """
        super().on_pruning_end()

        if self._is_ddp:
            dist.destroy_process_group(self._gloo_handle)

    def _broadcast_list_from_main(self, val: Any) -> Any:
        if not self._is_ddp:
            return val
        dist.broadcast_object_list(val, src=0, group=self._gloo_handle)
        return val


class MagnitudePruningParamsScorer(PruningParamsScorer):
    """
    Scores parameters based on their magnitude

    :param params: list of model Parameters to track and score
    """

    def score_parameters(self) -> List[Tensor]:
        """
        :return: List of Tensors the same shapes as the given Parameters where
            each Parameter's elements are scored by their magnitude (absolute value)
        """
        return [torch.abs(param.data) for param in self._params]

    @staticmethod
    def get_name() -> str:
        """
        :return: name of this pruning method
        """
        return "magnitude"


class MovementPruningParamsScorer(PruningParamsGradScorer):
    """
    Scores parameters based on their movement which is defined as
    movement_score = sum(-1.0 * W * dL/dW)

    Movement pruning introduced here: https://arxiv.org/abs/2005.07683

    :param params: list of model Parameters to track and score
    """

    def __init__(self, params: List[Parameter]):
        super().__init__(params)

        self._movement_scores = [
            param.data.new_zeros(param.data.shape).detach().requires_grad_(False)
            for param in self._params
        ]

    def score_parameters(self) -> List[Tensor]:
        """
        :return: List of Tensors the same shapes as the given Parameters where
            each Parameter's elements are scored by their weight times the direction
            of their gradient.
        """
        if not self._is_ddp:
            return self._movement_scores

        # move all movement scores to one device and combine
        scores_flat = [score.view(-1).to("cpu") for score in self._movement_scores]
        if self._is_main_proc:
            gather_list = [
                torch.zeros_like(scores_flat) for _ in range(dist.get_world_size())
            ]
            dist.gather(
                scores_flat, gather_list=gather_list, group=self._gloo_handle, dst=0
            )
            total_scores_flat = torch.sum(torch.stack(gather_list), dim=0)
        else:
            dist.gather(scores_flat, group=self._gloo_handle, dst=0)

        # broadcast total scores to all devices
        total_scores_flat = self._broadcast_list_from_main(
            [total_scores_flat if self._is_main_proc else None]
        )[0]

        # move total scores to correct device on each process
        score_idx = 0
        for idx, score in enumerate(self._movement_scores):
            next_idx = score_idx + score.numel()
            score.view(-1)[:] = total_scores_flat[score_idx:next_idx].to(score.device)
            score_idx = next_idx

        return self._movement_scores

    def pre_optim_step_update(self, masks: List[Tensor]):
        """
        Update movement scores based on the current Parameter weights and gradients

        :param masks: latest masks that are applied to these parameters
        """
        self.check_regen_param_vals()
        for idx, param in enumerate(self._params):
            if param.grad is not None and not torch.any(param.grad.isnan()):
                self._movement_scores[idx].add_(-0.01 * param.grad * param.data)

    def mask_update(self, masks: List[Tensor], mask_diffs: List[Tensor]):
        """
        Resets non main process scores after they have been recorded in the main
        process during the mask update

        :param masks: latest masks to be applied to these parameters
        :param mask_diffs: mask diff values returned by mask_difference for these
            masks that describe how these masks changed since the last update
        """
        if not self._is_main_proc:
            for score in self._movement_scores:
                score *= 0.0

    def check_regen_param_vals(self):
        """
        Check that movement scores are on the correct device and regenerate if not
        """
        for idx, param in enumerate(self._params):
            if self._params[idx].data.device != self._movement_scores[idx].device:
                self._movement_scores[idx] = (
                    torch.empty_like(self._params[idx].data)
                    .copy_(self._movement_scores[idx])
                    .detach()
                    .requires_grad_(False)
                )

    @staticmethod
    def get_name() -> str:
        """
        :return: name of this pruning method
        """
        return "movement"


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

        self._mfac_options = mfac_options or MFACOptions()
        self._unpruned_idxs = [None] * len(self._params)  # type: List[Tensor]
        self._grad_buffer = None  # type: Tensor
        self._grads = None  # placeholder for all grads across buffers
        self._buffer_idx = 0
        self._latest_h_inv_diag = None  # type: tuple

        # scale num_grads by number of DDP processes
        if self._is_ddp:
            world_size = dist.get_world_size()
            if isinstance(self._mfac_options, int):
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

        if any(param.grad is None for param in self._params):
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

    @staticmethod
    def get_name() -> str:
        """
        :return: name of this pruning method
        """
        return "MFAC"

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
            device=self._mfac_options.grads_device,
        )
        self._buffer_idx = 0


AVALIABLE_SCORER_CLASSES = [
    MagnitudePruningParamsScorer,
    MovementPruningParamsScorer,
    MFACPruningParamsScorer,
]  # type: List[PruningParamsScorer]


def create_pruning_param_scorer(
    params: List[Parameter],
    score_type: Union[str, MFACOptions],
) -> PruningParamsScorer:
    """
    :param params: List of Parameters for the created PruningParamsScorer to track
    :param score_type: String name of scoring type to use. Valid options are
        'magnitude', 'movement', or 'MFAC'. For MFAC pruning, passing in an MFACOptions
        object valid and is preferred.
    """
    scorer_name_to_constructor = {
        scorer.get_name(): scorer for scorer in AVALIABLE_SCORER_CLASSES
    }

    if isinstance(score_type, str):
        if score_type not in scorer_name_to_constructor:
            raise ValueError(
                f"Invalid score_type {score_type}. Valid score types include "
                f"{list(scorer_name_to_constructor.keys())}"
            )
        return scorer_name_to_constructor[score_type](params)

    if isinstance(score_type, MFACOptions):
        return MFACPruningParamsScorer(params, mfac_options=score_type)

    raise ValueError(
        f"Recieved unsupported type for score_type: {type(score_type)} "
        "expected string or MFACOptions object"
    )
