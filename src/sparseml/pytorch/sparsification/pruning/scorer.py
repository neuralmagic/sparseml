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
from typing import Any, Dict, List, Optional

import torch.distributed as dist
from torch import Tensor
from torch.nn import Parameter


__all__ = [
    "PruningParamsScorer",
    "PruningParamsGradScorer",
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


class PruningParamsGradScorer(PruningParamsScorer, ABC):
    """
    Abstract class for PruningParamsScorers that use gradients to score parameters.
    Adds extra abstraction for handling gradient sharing between parameters

    :param params: list of model Parameters to track and score
    :param dist_backend: to communicate gradients between processes
    """

    def __init__(
        self,
        params: List[Parameter],
        dist_backend: Optional[str] = None,
    ):
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
