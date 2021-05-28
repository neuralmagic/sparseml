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
from typing import List, Union

import torch
from torch import Tensor
from torch.nn import Parameter

from sparseml.pytorch.utils import MFACOptions, compute_hessian_inv


__all__ = [
    "PruningParamsScorer",
    "MagnitudePruningParamsScorer",
    "MovementPruningParamsScorer",
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

    def pre_optim_step_update(self):
        """
        Perform any required logic for tracking Parameter data and gradients before
            an Optimizer step is applied to the model.
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


class MovementPruningParamsScorer(PruningParamsScorer):
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
        return self._movement_scores

    def pre_optim_step_update(self):
        """
        Update movement scores based on the current Parameter weights and gradients
        """
        for idx, param in enumerate(self._params):
            if param.grad is not None:
                self._movement_scores[idx].add_(-0.01 * param.grad * param.data)

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


class MFACPruningParamsScorer(PruningParamsScorer):
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

        self._mfac_options = MFACOptions() or mfac_options
        self._unpruned_idxs = [None] * len(self._params)  # type: List[Tensor]
        self._grad_buffer = None  # type: Tensor
        self._buffer_idx = 0
        self._latest_h_inv_diag = None  # type: tuple

        self._setup_grad_buffer()

    def score_parameters(self) -> List[Tensor]:
        """
        :return: List of Tensors the same shapes as the given Parameters where
            each Parameter's elements are scored based on the optimal value
            given by the OBS method. For the approximated Hessian inverse matrix
            H^-1, scores will be W^2 / (2 * diag(H^-1))
        """
        if torch.any(torch.all(self._grad_buffer == 0.0, dim=1)):
            # if not all grads are captured, return magnitudes as scores
            return [torch.abs(param.data) for param in self._params]

        # gather non-pruned weights
        non_pruned_weights = torch.empty(self._grad_buffer.size(1)).to(
            self._grad_buffer.device
        )
        weights_idx = 0
        for idx, param in enumerate(self._params):
            indices = self._unpruned_idxs[idx]
            next_idx = weights_idx + indices.numel()
            non_pruned_weights[weights_idx:next_idx] = param.data.view(-1)[indices]
            weights_idx = next_idx

        # inverse hessian approximation
        h_inv = compute_hessian_inv(self._grad_buffer, self._mfac_options)
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
            param_score = torch.ones_like(param.data).detach().requires_grad_(False)
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

    def pre_optim_step_update(self):
        """
        Update the gradient buffer based on the current gradients
        """
        if any(param.grad is None for param in self._params):
            # only update buffer if all gradients are computed
            return

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
        # select weights that are about to be masked with 0s for unmasked weights
        weights_to_prune = torch.zeros(
            self._grad_buffer.size(1),
            device=self._grad_buffer.device,
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
        perturb = h_inv.mul(-1.0 * weights_to_prune / diag)
        weights_idx = 0

        # update weights by mapping to perturbation
        for idx, param in enumerate(self._params):
            indices = self._unpruned_idxs[idx]
            next_idx = weights_idx + indices.numel()
            param.view(-1)[self._unpruned_idxs[idx]] += perturb[
                weights_idx:next_idx
            ].to(param.device)
            weights_idx = next_idx

        self._latest_h_inv_diag = None  # clear h_inv
        self._setup_grad_buffer()  # reset grad buffer
        torch.cuda.empty_cache()  # release GPU memory

    def _setup_grad_buffer(self):
        total_nonzero = 0
        for idx, param in enumerate(self._params):
            self._unpruned_idxs[idx] = (
                param.view(-1).nonzero(as_tuple=False).reshape(-1)
            )
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


_SCORE_TYPE_TO_CONSTRUCTOR = {
    "magnitude": MagnitudePruningParamsScorer,
    "movement": MovementPruningParamsScorer,
    "MFAC": MFACPruningParamsScorer,
}


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
    if isinstance(score_type, str):
        if score_type not in _SCORE_TYPE_TO_CONSTRUCTOR:
            raise ValueError(
                f"Invalid score_type {score_type}. Valid score types include "
                f"{list(_SCORE_TYPE_TO_CONSTRUCTOR.keys())}"
            )
        return _SCORE_TYPE_TO_CONSTRUCTOR[score_type](params)
    if isinstance(score_type, MFACOptions):
        return MFACPruningParamsScorer(params, mfac_options=score_type)

    raise ValueError(
        f"Recieved unsupported type for score_type: {type(score_type)} "
        "expected string or MFACOptions object"
    )
