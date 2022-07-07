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

NOTE: this file is in the process of being phased out in favor of the
sparsification package. Once all references to mask utils in the optim
package are migrated, this file will be deleted
"""


from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Parameter


__all__ = [
    "AVALIABLE_SCORER_CLASSES",
    "PruningParamsScorer",
    "MagnitudePruningParamsScorer",
    "MovementPruningParamsScorer",
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
    :param dist_backend: to communicate gradients between processes
    """

    def __init__(self, params: List[Parameter], dist_backend: Optional[str] = None):
        super().__init__(params=params)

        self._is_ddp = dist.is_initialized()
        self._is_main_proc = not self._is_ddp or dist.get_rank() == 0

        # create group to broadcast gradients across processes
        self._dist_group = (
            dist.new_group(backend=dist_backend)
            if self._is_ddp and dist_backend is not None
            else None
        )

        self._pickle_exclude_params = ["_is_ddp", "_is_main_proc", "_dist_group"]

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

        if self._is_ddp and self._dist_group is not None:
            dist.destroy_process_group(self._dist_group)

    def _broadcast_list_from_main(self, val: Any) -> Any:
        if not self._is_ddp:
            return val
        dist.broadcast_object_list(val, src=0, group=self._dist_group)
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


AVALIABLE_SCORER_CLASSES = [
    MagnitudePruningParamsScorer,
    MovementPruningParamsScorer,
]  # type: List[PruningParamsScorer]


def create_pruning_param_scorer(
    params: List[Parameter], score_type: str
) -> PruningParamsScorer:
    """
    :param params: List of Parameters for the created PruningParamsScorer to track
    :param score_type: String name of scoring type to use. Valid options are
        'magnitude', or 'movement'
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

    raise ValueError(
        f"Recieved unsupported type for score_type: {type(score_type)} "
        "expected string"
    )
