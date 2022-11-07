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
Modifiers and support for structured (channel/filter) pruning
Thinning (removal of pruned channels) implemented by LayerThinningModifier
"""

from typing import Dict, List, Union

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Parameter

from sparseml.pytorch.sparsification.modifier import ModifierProp, PyTorchModifierYAML
from sparseml.pytorch.sparsification.pruning.modifier_pruning_magnitude import (
    GMPruningModifier,
)
from sparseml.pytorch.sparsification.pruning.scorer import PruningParamsGradScorer


__all__ = [
    "MovementPruningModifier",
    "MovementPruningParamsScorer",
]


@PyTorchModifierYAML()
class MovementPruningModifier(GMPruningModifier):
    """
    Gradually applies kernel sparsity to a given parameter or parameters from
    init_sparsity until final_sparsity is reached over a given amount of time
    and applied with an interpolated function for each step taken.

    Uses movement pruning to gradually mask parameter values.
    Movement pruning introduced here: https://arxiv.org/abs/2005.07683
    Pruning is unstructured by default, structure can be specified by mask_type.

    | Sample yaml:
    |   !MovementPruningModifier
    |       init_sparsity: 0.05
    |       final_sparsity: 0.8
    |       start_epoch: 0.0
    |       end_epoch: 10.0
    |       update_frequency: 1.0
    |       params: ["re:.*weight"]
    |       leave_enabled: True
    |       inter_func: cubic
    |       mask_type: unstructured

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
    :param mask_type: String to define type of sparsity (options: ['unstructured',
        'block']), List to define block shape of a parameters in and out
        channels, or a SparsityMaskCreator object. default is 'unstructured'
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
        mask_type: str = "unstructured",
    ):
        super(MovementPruningModifier, self).__init__(
            init_sparsity=init_sparsity,
            final_sparsity=final_sparsity,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            params=params,
            leave_enabled=leave_enabled,
            inter_func=inter_func,
            mask_type=mask_type,
        )

    def _get_scorer(self, params: List[Parameter]) -> PruningParamsGradScorer:
        """
        :param params: list of Parameters for scorer to track
        :return: param scorer object to be used by this pruning algorithm
        """
        return MovementPruningParamsScorer(params=params)

    @ModifierProp(serializable=False)
    def global_sparsity(self) -> bool:
        """
        :return: True for global magnitude pruning, False for
            layer-wise. [DEPRECATED] - use GlobalMagnitudePruningModifier
            for global magnitude pruning and MagnitudePruningModifier for layer-wise
        """
        return self._global_sparsity


class MovementPruningParamsScorer(PruningParamsGradScorer):
    """
    Scores parameters based on their movement which is defined as
    movement_score = sum(-1.0 * W * dL/dW)

    Movement pruning introduced here: https://arxiv.org/abs/2005.07683

    :param params: list of model Parameters to track and score
    """

    def __init__(self, params: List[Parameter]):
        super().__init__(params, dist_backend="gloo")

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
                scores_flat, gather_list=gather_list, group=self._dist_group, dst=0
            )
            total_scores_flat = torch.sum(torch.stack(gather_list), dim=0)
        else:
            dist.gather(scores_flat, group=self._dist_group, dst=0)

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
