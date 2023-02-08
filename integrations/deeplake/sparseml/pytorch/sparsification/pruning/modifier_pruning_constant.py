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

from typing import List, Union

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from sparseml.pytorch.sparsification.modifier import (
    ModifierProp,
    PyTorchModifierYAML,
    ScheduledModifier,
)
from sparseml.pytorch.sparsification.pruning.mask_creator import PruningMaskCreator
from sparseml.pytorch.sparsification.pruning.modifier_pruning_base import (
    BasePruningModifier,
)
from sparseml.pytorch.utils import get_prunable_layers, tensor_sparsity
from sparseml.sparsification import (
    ConstantPruningModifier as BaseConstantPruningModifier,
)


__all__ = [
    "ConstantMaskCreator",
    "ConstantPruningModifier",
]


class ConstantMaskCreator(PruningMaskCreator):
    """
    Class for creating sparsity masks that only mask already pruned parameters.
    i.e. if the value of a paraemeter is 0 it will be masked, otherwise it will
        remain unmasked
    """

    def create_sparsity_masks(
        self,
        tensors: List[Tensor],
        target: Union[float, List[float]],
        global_sparsity: bool = False,
    ) -> List[Tensor]:
        """
        :param tensors: tensors to generate constant masks for
        :param target: not used for constant pruning
        :param global_sparsity: not used for constant pruning
        :return: list of masks derived from pruned values of each of the given tensors
        """
        return [torch.ne(tensor, 0.0).type(tensor.type()) for tensor in tensors]


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

    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param update_frequency: Ignored for this modifier
    :param params: A list of full parameter names or regex patterns of names to apply
        pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
        will match to all parameters. __ALL_PRUNABLE__ will match to all ConvNd
        and Linear layers' weights
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
    ):
        super(ConstantPruningModifier, self).__init__(
            params=params,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            end_comparator=-1,
            update_frequency=-1,
            allow_reintroduction=False,
            leave_enabled=False,
            parent_class_kwarg_names=["params"],
        )

    def _get_mask_creator(
        self, param_names: List[str], params: List[Parameter]
    ) -> PruningMaskCreator:
        """
        :param names: full names of parameters to be pruned
        :param params: list of Parameters to be masked
        :return: mask creator object to be used by this pruning algorithm
        """
        return ConstantMaskCreator()

    def _get_scorer(self, *args, **kwargs):
        """
        :return: None, no scorer is used, defaults to using raw parameter values
        """
        return None

    def get_applied_sparsity_for_epoch(self, *args, **kwargs):
        """
        :return: None, sparsity is set by the existing levels
        """
        return None

    @ModifierProp(serializable=False)
    def leave_enabled(self) -> bool:
        """
        :return: True to continue masking the weights after end_epoch,
        False to stop masking. Should be set to False if exporting the result
        immediately after or doing some other prune.
        """
        return self._leave_enabled

    @ModifierProp(serializable=False)
    def global_sparsity(self) -> bool:
        """
        :return: value of global_sparsity that is passed to mask_creator methods
        """
        return self._global_sparsity
