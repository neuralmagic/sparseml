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
Modifiers classes related to magnitude pruning
"""

import logging
from typing import Dict, List, Union

import torch
from torch import Tensor
from torch.nn import Parameter

from sparseml.pytorch.sparsification.modifier import ModifierProp, PyTorchModifierYAML
from sparseml.pytorch.sparsification.pruning.mask_creator import (
    PruningMaskCreator,
    get_mask_creator_default,
)
from sparseml.pytorch.sparsification.pruning.modifier_pruning_base import (
    BaseGradualPruningModifier,
)
from sparseml.pytorch.sparsification.pruning.scorer import PruningParamsScorer
from sparseml.sparsification import GMPruningModifier as BaseGMPruningModifier


__all__ = [
    "MagnitudePruningParamsScorer",
    "MagnitudePruningModifier",
    "GMPruningModifier",
    "GlobalMagnitudePruningModifier",
]


_LOGGER = logging.getLogger(__name__)


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


@PyTorchModifierYAML()
class GMPruningModifier(BaseGradualPruningModifier, BaseGMPruningModifier):
    """
    Gradually applies kernel sparsity to a given parameter or parameters from
    init_sparsity until final_sparsity is reached over a given amount of time
    and applied with an interpolated function for each step taken.

    Applies based on magnitude pruning unless otherwise specified by mask_type.

    | Sample yaml:
    |   !GMPruningModifier
    |       init_sparsity: 0.05
    |       final_sparsity: 0.8
    |       start_epoch: 0.0
    |       end_epoch: 10.0
    |       update_frequency: 1.0
    |       params: ["re:.*weight"]
    |       leave_enabled: True
    |       inter_func: cubic
    |       mask_type: unstructured

    :param init_sparsity: initial sparsity for each param to start with at
        start_epoch. __FROM_PARAM__ will set initial sparsity for each param to
        the existing sparsity level in that param
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
    :param mask_type: String to define type of sparsity to apply. May be 'unstructred'
        for unstructured pruning or 'block4' for four block pruning or a list of two
        integers for a custom block shape. Default is 'unstructured'
    :param global_sparsity: set True to use global magnitude pruning, False for
        layer-wise. Default is False. [DEPRECATED] - use GlobalMagnitudePruningModifier
        for global magnitude pruning and MagnitudePruningModifier for layer-wise
    :param phased: NO LONGER SUPPORTED - former parameter for AC/DC pruning. Will raise
        an exception if set to True. Use ACDCPruningModifier for AC/DC pruning
    :param score_type: NO LONGER SUPPORTED - former parameter for using different
        sparsification algorithms, will raise an exception if set to the non default
        value
    """

    def __init__(
        self,
        init_sparsity: Union[float, str],
        final_sparsity: Union[float, Dict[float, List[str]]],
        start_epoch: float,
        end_epoch: float,
        update_frequency: float,
        params: Union[str, List[str]],
        leave_enabled: bool = True,
        inter_func: str = "cubic",
        mask_type: str = "unstructured",
        global_sparsity: bool = False,
        phased: bool = False,
        score_type: str = "magnitude",
    ):
        self._check_deprecated_params(global_sparsity, phased, score_type)

        super(GMPruningModifier, self).__init__(
            params=params,
            init_sparsity=init_sparsity,
            final_sparsity=final_sparsity,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            inter_func=inter_func,
            mask_type=mask_type,
            leave_enabled=leave_enabled,
            global_sparsity=global_sparsity,
            end_comparator=-1,
            allow_reintroduction=False,
            parent_class_kwarg_names=[
                "init_sparsity",
                "final_sparsity",
                "params",
                "leave_enabled",
                "mask_type",
                "inter_func",
            ],
        )

    def _get_mask_creator(
        self, param_names: List[str], params: List[Parameter]
    ) -> PruningMaskCreator:
        """
        :param names: full names of parameters to be pruned
        :param params: list of parameters to be masked
        :return: mask creator object to be used by this pruning algorithm
        """
        return get_mask_creator_default(self.mask_type)

    def _get_scorer(self, params: List[Parameter]) -> PruningParamsScorer:
        """
        :param params: list of Parameters for scorer to track
        :return: param scorer object to be used by this pruning algorithm
        """
        return MagnitudePruningParamsScorer(params)

    @ModifierProp()
    def global_sparsity(self) -> bool:
        """
        :return: True for global magnitude pruning, False for
            layer-wise. [DEPRECATED] - use GlobalMagnitudePruningModifier
            for global magnitude pruning and MagnitudePruningModifier for layer-wise
        """
        return self._global_sparsity

    def _check_deprecated_params(
        self,
        global_sparsity: bool,
        phased: bool,
        score_type: str,
    ):
        if self.__class__.__name__ == "GMPruningModifier" and global_sparsity is True:
            _LOGGER.warning(
                "Use of global_sparsity parameter in GMPruningModifier is now "
                "deprecated. Use GlobalMagnitudePruningModifier instead for global "
                "magnitude pruning"
            )
        if phased:
            raise ValueError(
                f"Use of phased=True in {self.__class__.__name__} is no longer "
                "supported use the ACDCPruningModifier for phased (AC/DC) pruning"
            )
        if score_type != "magnitude":
            raise ValueError(
                "use of score_type to specify a sparsification algorithm is no longer "
                "supported. Use the specific pruning modifier for the desired "
                f"sparsification algorithm instead. Found score_type={score_type}"
            )


@PyTorchModifierYAML()
class MagnitudePruningModifier(GMPruningModifier):
    """
    Gradually applies kernel sparsity to a given parameter or parameters from
    init_sparsity until final_sparsity is reached over a given amount of time
    and applied with an interpolated function for each step taken.

    Applies based on magnitude pruning unless otherwise specified by mask_type.

    | Sample yaml:
    |   !MagnutidePruningModifier
    |       init_sparsity: 0.05
    |       final_sparsity: 0.8
    |       start_epoch: 0.0
    |       end_epoch: 10.0
    |       update_frequency: 1.0
    |       params: ["re:.*weight"]
    |       leave_enabled: True
    |       inter_func: cubic
    |       mask_type: unstructured

    :param init_sparsity: initial sparsity for each param to start with at
        start_epoch. __FROM_PARAM__ will set initial sparsity for each param to
        the existing sparsity level in that param
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
    :param mask_type: String to define type of sparsity to apply. May be 'unstructred'
        for unstructured pruning or 'block4' for four block pruning or a list of two
        integers for a custom block shape. Default is 'unstructured'
    """

    def __init__(
        self,
        init_sparsity: Union[float, str],
        final_sparsity: Union[float, Dict[float, List[str]]],
        start_epoch: float,
        end_epoch: float,
        update_frequency: float,
        params: Union[str, List[str]],
        leave_enabled: bool = True,
        inter_func: str = "cubic",
        mask_type: str = "unstructured",
    ):
        super(MagnitudePruningModifier, self).__init__(
            params=params,
            init_sparsity=init_sparsity,
            final_sparsity=final_sparsity,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            inter_func=inter_func,
            mask_type=mask_type,
            leave_enabled=leave_enabled,
            global_sparsity=False,
        )

    @ModifierProp(serializable=False)
    def global_sparsity(self) -> bool:
        """
        :return: True for global magnitude pruning, False for
            layer-wise. [DEPRECATED] - use GlobalMagnitudePruningModifier
            for global magnitude pruning and MagnitudePruningModifier for layer-wise
        """
        return self._global_sparsity


@PyTorchModifierYAML()
class GlobalMagnitudePruningModifier(GMPruningModifier):
    """
    Gradually applies kernel sparsity to a given parameter or parameters from
    init_sparsity until final_sparsity is reached over a given amount of time
    and applied with an interpolated function for each step taken.

    Applies based on magnitude pruning unless otherwise specified by mask_type.

    | Sample yaml:
    |   !GlobalMagnitudePruningModifier
    |       init_sparsity: 0.05
    |       final_sparsity: 0.8
    |       start_epoch: 0.0
    |       end_epoch: 10.0
    |       update_frequency: 1.0
    |       params: ["re:.*weight"]
    |       leave_enabled: True
    |       inter_func: cubic
    |       mask_type: unstructured

    :param init_sparsity: initial sparsity for each param to start with at
        start_epoch. __FROM_PARAM__ will set initial sparsity for each param to
        the existing sparsity level in that param
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
    :param mask_type: String to define type of sparsity to apply. May be 'unstructred'
        for unstructured pruning or 'block4' for four block pruning or a list of two
        integers for a custom block shape. Default is 'unstructured'
    """

    def __init__(
        self,
        init_sparsity: Union[float, str],
        final_sparsity: Union[float, Dict[float, List[str]]],
        start_epoch: float,
        end_epoch: float,
        update_frequency: float,
        params: Union[str, List[str]],
        leave_enabled: bool = True,
        inter_func: str = "cubic",
        mask_type: str = "unstructured",
    ):
        super(GlobalMagnitudePruningModifier, self).__init__(
            params=params,
            init_sparsity=init_sparsity,
            final_sparsity=final_sparsity,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            inter_func=inter_func,
            mask_type=mask_type,
            leave_enabled=leave_enabled,
            global_sparsity=True,
        )

    @ModifierProp(serializable=False)
    def global_sparsity(self) -> bool:
        """
        :return: True for global magnitude pruning, False for
            layer-wise. [DEPRECATED] - use GlobalMagnitudePruningModifier
            for global magnitude pruning and MagnitudePruningModifier for layer-wise
        """
        return self._global_sparsity
