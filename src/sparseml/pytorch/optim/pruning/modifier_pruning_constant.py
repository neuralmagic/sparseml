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
Modifiers classes related to holding sparsity level constant for finetuning or
transfer learning
"""

from typing import Dict, List, Union

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from sparseml.pytorch.optim.modifier import PyTorchModifierYAML, ScheduledModifier
from sparseml.pytorch.optim.pruning.mask_creator import (
    PruningMaskCreator,
    UnstructuredPruningMaskCreator,
)
from sparseml.pytorch.optim.pruning.modifier_pruning_base import BasePruningModifier
from sparseml.pytorch.optim.pruning.scorer import PruningParamsScorer
from sparseml.pytorch.utils import get_prunable_layers, tensor_sparsity
from sparseml.sparsification import (
    ConstantPruningModifier as BaseConstantPruningModifier,
)
from sparseml.utils import ALL_TOKEN


__all__ = ["ConstantPruningModifier", "IdentityPruningParamsScorer"]


class IdentityPruningParamsScorer(PruningParamsScorer):
    """
    Scores parameters based on their current value

    :param params: list of model Parameters to track and score
    """

    def score_parameters(self) -> List[Tensor]:
        """
        :return: List of Tensors the same shapes as the given Parameters where
            each Parameter's elements are scored by their magnitude (absolute value)
        """
        return [param.data for param in self._params]


@PyTorchModifierYAML()
class ConstantPruningModifier(BasePruningModifier, BaseConstantPruningModifier):
    """
    Holds the sparsity level and shape for a given parameter(s) constant while training.
    Useful for transfer learning use cases.

    | Sample yaml:
    |   !ConstantPruningModifier
    |       start_epoch: 0.0
    |       end_epoch: 10.0
    |       params: ['re:.*weight']
    |       log_types: __ALL__

    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param update_frequency: Ignored for this modifier
    :param params: A list of full parameter names or regex patterns of names to apply
        pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
        will match to all parameters. __ALL_PRUNABLE__ will match to all ConvNd
        and Linear layers' weights
    :param log_types: The loggers to allow the learning rate to be logged to,
        default is __ALL__
    """

    @staticmethod
    def from_sparse_model(model: Module) -> List[ScheduledModifier]:
        """
        Create constant ks modifiers for all prunable params in the given model
        (conv, linear) that have been artificially sparsified (sparsity > 40%).
        Useful for transfer learning from a pruned model.

        :param model: the model to create constant ks modifiers for
        :return: the list of created constant ks modifiers
        """
        prunable = get_prunable_layers(model)
        modifiers = []

        for name, layer in prunable:
            weight = getattr(layer, "weight")
            sparsity = tensor_sparsity(weight)

            if sparsity > 0.1:  # set at 10% sparsity to be threshold for intentional
                modifiers.append(
                    ConstantPruningModifier(params=["{}.{}".format(name, "weight")])
                )

        return modifiers

    def __init__(
        self,
        params: Union[str, List[str]],
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
        update_frequency: float = -1.0,
        log_types: Union[str, List[str]] = ALL_TOKEN,
    ):
        super(ConstantPruningModifier, self).__init__(
            params=params,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            end_comparator=-1,
            update_frequency=-1,
            log_types=log_types,
            allow_reintroduction=False,
        )

    def _get_mask_creator(self) -> PruningMaskCreator:
        """
        :return: mask creator object to be used by this pruning algorithm
        """
        return UnstructuredPruningMaskCreator()

    def _get_scorer(self, params: List[Parameter]) -> PruningParamsScorer:
        """
        :param params: list of Parameters for scorer to track
        :return: param scorer object to be used by this pruning algorithm
        """
        return IdentityPruningParamsScorer(params)

    def get_applied_sparsity_for_epoch(self, *args, **kwargs):
        """
        :return: None, sparsity is set by the existing levels
        """
        return None

    def check_mask_update(
        self, module: Module, epoch: float, steps_per_epoch: int, **kwargs
    ):
        """
        Override normal pruning update to only update masks on start and end
        to keep current sparsity level constant

        :param module: module to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        if self.start_pending(epoch, steps_per_epoch):
            self._module_masks.set_param_masks_from_weights()
            self._module_masks.enabled = True

        if self.end_pending(epoch, steps_per_epoch):
            self._module_masks.set_param_masks_from_weights()
            self._module_masks.enabled = False
