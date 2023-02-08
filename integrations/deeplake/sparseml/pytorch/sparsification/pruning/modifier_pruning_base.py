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
Base classes for creating modifiers for pruning algorithms
"""

import math
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

from torch import Tensor
from torch.nn import Module, Parameter
from torch.optim.optimizer import Optimizer

from sparseml.optim.modifier import BaseModifier
from sparseml.pytorch.optim.analyzer_pruning import ModulePruningAnalyzer
from sparseml.pytorch.sparsification.modifier import (
    ModifierProp,
    ScheduledModifier,
    ScheduledUpdateModifier,
)
from sparseml.pytorch.sparsification.pruning.mask_creator import PruningMaskCreator
from sparseml.pytorch.sparsification.pruning.mask_params import ModuleParamPruningMask
from sparseml.pytorch.sparsification.pruning.scorer import PruningParamsScorer
from sparseml.pytorch.utils import (
    NamedLayerParam,
    get_named_layers_and_params_by_regex,
    get_prunable_layers,
    tensor_sparsity,
)
from sparseml.pytorch.utils.logger import LoggerManager
from sparseml.sparsification import SparsificationTypes
from sparseml.utils import (
    ALL_PRUNABLE_TOKEN,
    ALL_TOKEN,
    FROM_PARAM_TOKEN,
    interpolate,
    validate_str_iterable,
)


__all__ = [
    "BasePruningModifier",
    "BaseGradualPruningModifier",
]


class BasePruningModifier(ABC, ScheduledUpdateModifier):
    """
    Base class for pruning modifiers that create masks for given params

    Lifecycle:
        | initialize()
        |    params         <- _create_named_layers_and_params()
        |    mask_creator   <- _get_mask_creator()
        |    scorer         <- _get_scorer()
        |    module_masks   <- ModuleParamPruningMask(params, mask_creator, scorer)
        |    initialize_extras  <- perform any extra initialization steps
        |
        | update()
        |    applied_sparsity   <- get_applied_sparsity_for_epoch()
        |    module_masks.update_param_masks(applied_sparsity)
        |
        | optimizer_pre_step()
        | optimizer.step()
        | optimizer_post_step()
        |
        | finalize()


    :param params: A list of full parameter names or regex patterns of names to apply
        pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
        will match to all parameters. __ALL_PRUNABLE__ will match to all ConvNd
        and Linear layers' weights
    :param start_epoch: The epoch to start the modifier at
    :param min_start: The minimum acceptable value for start_epoch, default -1
    :param end_epoch: The epoch to end the modifier at
    :param min_end: The minimum acceptable value for end_epoch, default 0
    :param end_comparator: integer value representing how the end_epoch should be
        compared to start_epoch.
        if == None, then end_epoch can only be set to what its initial value was.
        if == -1, then end_epoch can be -1, equal, or greater than start_epoch.
        if == 0, then end_epoch can be equal to or greater than start_epoch.
        if == 1, then end_epoch can only be greater than start_epoch.
    :param update_frequency: The number of epochs or fraction of epochs to
            update at between start and end
    :param min_frequency: The minimum acceptable value for update_frequency,
        default -1
    :param global_sparsity: set True to pass global_sparsity as True to mask
        creator methods. Default is False
    :param allow_reintroduction: if True, gradients and params will not be masked
        between forward passes. Default is False
    :param parent_class_kwarg_names: a list of args which indicates the subset of kwargs
        to pass to super.__init__. Resulting kwargs will be the set intersect of
        parent_class_kwarg_names and the initial kwargs. A value of None will preserve
        the original kwargs, whereas an empty list will yield empty kwargs.
        This param is included to avoid collisions between multiple inheritance
        requirements and SparseML requirements to not pass kwargs to the BaseObject
        class.
    :param leave_enabled: True to continue masking the weights after end_epoch,
        False to stop masking. Should be set to False if exporting the result
        immediately after or doing some other prune. Default is False
    """

    def __init__(
        self,
        params: Union[str, List[str]],
        start_epoch: float = -1.0,
        min_start: float = -1.0,
        end_epoch: float = -1.0,
        min_end: float = -1.0,
        end_comparator: Union[int, None] = 0,
        update_frequency: float = -1.0,
        min_frequency: float = -1.0,
        global_sparsity: bool = False,
        allow_reintroduction: bool = False,
        leave_enabled: bool = False,
        parent_class_kwarg_names: Optional[List[str]] = None,
        **kwargs,
    ):
        if parent_class_kwarg_names is not None:
            # filter kwargs only for ones that should be propagated
            # parent_class_kwarg_names = ["params", "init_sparsity", "iterpolation",...]
            kwargs = {k: v for k, v in kwargs.items() if k in parent_class_kwarg_names}
            if "params" in parent_class_kwarg_names:
                kwargs["params"] = params
        super().__init__(
            start_epoch=start_epoch,
            min_start=min_start,
            end_epoch=end_epoch,
            min_end=min_end,
            end_comparator=end_comparator,
            update_frequency=update_frequency,
            min_frequency=min_frequency,
            **kwargs,
        )
        self._params = validate_str_iterable(
            params, "{} for params".format(self.__class__.__name__)
        )
        self._module_masks = None  # type: Optional[ModuleParamPruningMask]
        self._analyzers = None  # type: Optional[List[ModulePruningAnalyzer]]

        self._scorer = None  # type: PruningParamsScorer
        self._mask_creator = None  # type: PruningMaskCreator

        self._global_sparsity = global_sparsity
        self._allow_reintroduction = allow_reintroduction
        self._leave_enabled = leave_enabled

        self._applied_sparsity = None
        self._pre_step_completed = False
        self._sparsity_applied = False

    @BaseModifier.sparsification_types.getter
    def sparsification_types(self) -> List[SparsificationTypes]:
        """
        :return: the sparsification types this modifier instance will apply
        """
        return [SparsificationTypes.pruning]

    @abstractmethod
    def _get_mask_creator(
        self, param_names: List[str], params: List[Parameter]
    ) -> PruningMaskCreator:
        """
        :param names: full names of parameters to be pruned
        :param params: list of Parameters to be masked
        :return: mask creator object to be used by this pruning algorithm
        """
        raise NotImplementedError()

    @abstractmethod
    def _get_scorer(self, params: List[Parameter]) -> PruningParamsScorer:
        """
        :param params: list of Parameters for scorer to track
        :return: param scorer object to be used by this pruning algorithm
        """
        raise NotImplementedError()

    @abstractmethod
    def get_applied_sparsity_for_epoch(
        self, epoch: float, steps_per_epoch: int
    ) -> Union[float, List[float]]:
        """
        :param epoch: curent epoch
        :param steps_per_epoch: number of steps per epoch
        :return: sparsity level that should be applied at the given epoch. If parameters
            should be set to different sparsities, should return a list of those values
            in the order the parameters appear in the mask manager for this object
        """
        raise NotImplementedError()

    @ModifierProp()
    def params(self) -> Union[str, List[str], None]:
        """
        :return: A list of full parameter names or regex patterns of names to apply
            pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
            will match to all parameters. __ALL_PRUNABLE__ will match to all ConvNd
            and Linear layers' weights
        """
        return self._params

    @ModifierProp()
    def leave_enabled(self) -> bool:
        """
        :return: True to continue masking the weights after end_epoch,
        False to stop masking. Should be set to False if exporting the result
        immediately after or doing some other prune.
        """
        return self._leave_enabled

    @ModifierProp()
    def global_sparsity(self) -> bool:
        """
        :return: value of global_sparsity that is passed to mask_creator methods
        """
        return self._global_sparsity

    @property
    def module_masks(self) -> Optional[ModuleParamPruningMask]:
        """
        :return: The mask instances corresponding to the desired params passed in
            to the current pruning modifier that contain the masking information
        """
        return self._module_masks

    @property
    def analyzers(self) -> Optional[List[ModulePruningAnalyzer]]:
        """
        :return: The analyzer instances corresponding to the desired params passed in
            to the current pruning modifier that contain the analyzing information
        """
        return self._analyzers

    @property
    def mask_creator(self) -> Optional[PruningMaskCreator]:
        """
        :return: mask creator object used by this pruning algorithm
        """
        return self._mask_creator

    @property
    def scorer(self) -> Optional[PruningParamsScorer]:
        """
        :return: param scorer object used by this pruning algorithm
        """
        return self._scorer

    @property
    def allow_reintroduction(self) -> bool:
        """
        :return: True if gradients and params are not masked outside of forward passes
        """
        return self._allow_reintroduction

    @property
    def applied_sparsity(self) -> float:
        """
        :return: latest sparsity value applied
        """
        return self._applied_sparsity

    def is_oneshot(self, epoch: float, steps_per_epoch: int) -> bool:
        """
        :return: True if modifier called in a one-shot manner
        """
        return steps_per_epoch == 1 and math.isinf(epoch)

    def initialize(
        self,
        module: Module,
        epoch: float = 0,
        loggers: Optional[LoggerManager] = None,
        **kwargs,
    ):
        """
        Grab the params and apply if epoch in range to control pruning for.

        :param module: the PyTorch model/module to modify
        :param epoch: The epoch to initialize the modifier and module at.
            Defaults to 0 (start of the training process)
        :param loggers: Optional list of loggers to log the modification process to
        :param kwargs: Optional kwargs to support specific arguments
            for individual modifiers.
        """
        super().initialize(module, epoch, loggers, **kwargs)
        named_layers_and_params = self._create_named_layers_and_params(module)
        layers = [nlp.layer for nlp in named_layers_and_params]
        param_names = [nlp.param_name for nlp in named_layers_and_params]
        layer_names = [nlp.layer_name for nlp in named_layers_and_params]

        # initialize mask_creator and scorer
        params = [
            getattr(layer, param_name) for layer, param_name in zip(layers, param_names)
        ]
        full_param_names = [
            f"{layer_name}.{param_name}"
            for layer_name, param_name in zip(layer_names, param_names)
        ]
        self._mask_creator = self._get_mask_creator(full_param_names, params)
        self._scorer = self._get_scorer(params)

        self._module_masks = self._create_pruning_mask(layers, layer_names, param_names)
        self._analyzers = self._create_analyzers(layers, layer_names, param_names)

        if len(self._analyzers) == 0:
            raise ValueError(
                "Could not find any params matching {} in {}".format(
                    self._params, self.__class__.__name__
                )
            )

        self.initialize_extras(module)

        if self.is_oneshot(epoch, steps_per_epoch=1):
            self.check_mask_update(module, epoch, steps_per_epoch=1, **kwargs)

    def initialize_extras(self, module: Module):
        """
        Perform any extra setup for pruning modifier after params have been parsed,
        and mask creator and scorers initialized

        :param module: module to be pruned
        """
        pass

    @ScheduledModifier.log_call
    def update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Update to enable and disable the mask when chosen.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().update(module, optimizer, epoch, steps_per_epoch)
        self.check_mask_update(module, epoch, steps_per_epoch)

    def check_mask_update(
        self,
        module: Module,
        epoch: float,
        steps_per_epoch: int,
        recomputation_sparsity: Optional[float] = None,
        **kwargs,
    ):
        """
        Update mask values if necessary

        :param module: module to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        started = self.started
        if self.start_pending(epoch, steps_per_epoch):
            self._module_masks.enabled = True
            started = True

        if not self._pre_step_completed:
            # do pre optim step before mask update on update steps
            self._module_masks.pre_optim_step_update()
            self._pre_step_completed = True

        if started:
            # get sparsity level to be applied
            self._applied_sparsity = self.get_applied_sparsity_for_epoch(
                epoch, steps_per_epoch
            )

            self._module_masks.update_param_masks(
                target=recomputation_sparsity or self._applied_sparsity
            )
            self._sparsity_applied = True

        if self.end_pending(epoch, steps_per_epoch):
            self._module_masks.pruning_end(self._leave_enabled)

    def log_update(
        self,
        module: Module,
        optimizer: Optimizer,
        epoch: float,
        steps_per_epoch: int,
    ):
        """
        Check whether to log an update for the learning rate of the modifier.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().log_update(module, optimizer, epoch, steps_per_epoch)

        for layer_sparsity in self._analyzers:
            if isinstance(layer_sparsity, ModulePruningAnalyzer):
                layer_sparsity = (
                    layer_sparsity.tag,
                    layer_sparsity.param_sparsity.item(),
                )
                self.log_scalar(
                    tag=f"ParamPruning/{layer_sparsity[0]}",
                    value=layer_sparsity[1],
                    epoch=epoch,
                    steps_per_epoch=steps_per_epoch,
                )

    def optimizer_pre_step(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Performs any tracking or updates before the optimizer step is applied. Useful
        for tracking gradient values between backwards pass and optimizer step if
        optimizer clips gradients

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().optimizer_pre_step(module, optimizer, epoch, steps_per_epoch)

        if not self._pre_step_completed:
            self._module_masks.pre_optim_step_update()
        self._pre_step_completed = False

    def optimizer_post_step(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Reapply the mask after the optimizer step in case the optimizer
        has momentum that may have moved weights from 0.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().optimizer_post_step(module, optimizer, epoch, steps_per_epoch)

        if not self._allow_reintroduction:
            # be sure to apply mask again after optimizer update because
            # weights may have changed (optimizer with momentum, not masking gradient)
            self._module_masks.apply()

        self._sparsity_applied = False

    def finalize(
        self, module: Optional[Module] = None, reset_loggers: bool = True, **kwargs
    ):
        """
        Cleans up any remaining hooks

        :param module: The model/module to finalize the modifier for.
            Marked optional so state can still be cleaned up on delete,
            but generally should always be passed in.
        :param reset_loggers: True to remove any currently attached loggers (default),
            False to keep the loggers attached.
        :param kwargs: Optional kwargs to support specific arguments
            for individual modifiers.
        """
        super().finalize(module, reset_loggers, **kwargs)
        self._module_masks.apply()
        self._module_masks.enabled = False
        self._module_masks = None
        self._analyzers = None

    def state_dict(self) -> Dict[str, Tensor]:
        """
        :return: PyTorch state dictionary to store any variables from this modifier.
            The mapping is param_name -> mask
        """
        return OrderedDict(
            zip(self._module_masks.names, self._module_masks.param_masks)
        )

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True):
        """
        Loads the given state dict into this object's modifiers

        :param state_dict: dictionary object as generated by this object's state_dict
            function
        :param strict: Ignored for this modifier, everything is treated as strict
        :raises IndexError: If any keys in the state dict do not correspond to a valid
            index for this manager and strict=True
        """
        if not self.initialized:
            raise RuntimeError("Cannot load state dict for an uninitialized modifier")

        mask_names = {key for key in self._module_masks.names}
        state_dict_keys = {key for key in state_dict.keys()}
        diff = mask_names.symmetric_difference(state_dict_keys)
        if diff and strict:
            raise IndexError(
                f"Found extra keys: {state_dict_keys - mask_names} "
                f"and missing keys: {mask_names - state_dict_keys}"
            )

        self._module_masks.set_param_masks(
            [state_dict[name] for name in self._module_masks.names]
        )

    def _check_params_match(self, token: Union[str, List[str]]):
        if isinstance(token, str):
            return token in self._params or token == self._params

        if isinstance(self._params, str):
            return self._params in token

        return len(set(token).intersection(set(self._params))) > 0

    def _create_named_layers_and_params(self, module: Module) -> List[NamedLayerParam]:
        if self._check_params_match(ALL_TOKEN):
            param_names = ["re:.*"]
        elif self._check_params_match(ALL_PRUNABLE_TOKEN):
            param_names = [
                name + ".weight" for (name, _) in get_prunable_layers(module)
            ]
        else:
            param_names = self._params

        return get_named_layers_and_params_by_regex(
            module,
            param_names,
            params_strict=True,
        )

    def _create_pruning_mask(
        self, layers: List[Module], layer_names: List[str], param_names: List[str]
    ) -> ModuleParamPruningMask:
        return ModuleParamPruningMask(
            layers,
            mask_creator=self._mask_creator,
            scorer=self._scorer,
            param_names=param_names,
            layer_names=layer_names,
            global_sparsity=self._global_sparsity,
            allow_reintroduction=self._allow_reintroduction,
        )

    def _create_analyzers(
        self, layers: List[Module], layer_names: List[str], param_names: List[str]
    ):
        return [
            ModulePruningAnalyzer(layer, layer_name, param_name)
            for (layer, layer_name, param_name) in zip(layers, layer_names, param_names)
        ]


class BaseGradualPruningModifier(BasePruningModifier):
    """
    Base class for gradual pruners that start and end at given sparsities and follow
    an interpolation function. Subclasses must still implement intializers for mask
    creator and scorer. get_applied_sparsity_for_epoch is implemented based on the
    interpolation function

    :param params: A list of full parameter names or regex patterns of names to apply
        pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
        will match to all parameters. __ALL_PRUNABLE__ will match to all ConvNd
        and Linear layers' weights
    :param init_sparsity: initial sparsity for each param to start with at
        start_epoch. __FROM_PARAM__ will set initial sparsity for each param to
        the existing sparsity level in that param
    :param final_sparsity: the final sparsity for the param to end with at end_epoch.
        Can also be a Dict of final sparsity values to a list of parameters to apply
        them to. If given a Dict, then params must be set to [] and the params to
        be pruned will be read from the final_sparsity Dict
    :param inter_func: the type of interpolation function to use:
        [linear, cubic, inverse_cubic]
    :param start_epoch: The epoch to start the modifier at
    :param min_start: The minimum acceptable value for start_epoch, default -1
    :param end_epoch: The epoch to end the modifier at
    :param min_end: The minimum acceptable value for end_epoch, default 0
    :param end_comparator: integer value representing how the end_epoch should be
        compared to start_epoch.
        if == None, then end_epoch can only be set to what its initial value was.
        if == -1, then end_epoch can be -1, equal, or greater than start_epoch.
        if == 0, then end_epoch can be equal to or greater than start_epoch.
        if == 1, then end_epoch can only be greater than start_epoch.
    :param update_frequency: The number of epochs or fraction of epochs to
            update at between start and end
    :param min_frequency: The minimum acceptable value for update_frequency,
        default -1
    :param global_sparsity: set True to pass global_sparsity as True to mask
        creator methods. Default is False
    :param allow_reintroduction: if True, gradients and params will not be masked
        between forward passes. Default is False
    :param parent_class_kwarg_names: a list of args which indicates the subset of kwargs
        to pass to super.__init__. Resulting kwargs will be the set intersect of
        parent_class_kwarg_names and the initial kwargs. A value of None will preserve
        the original kwargs, whereas an empty list will yield empty kwargs.
        This param is included to avoid collisions between multiple inheritance
        requirements and SparseML requirements to not pass kwargs to the BaseObject
        class.
    """

    def __init__(
        self,
        params: Union[str, List[str]],
        init_sparsity: Union[float, str],
        final_sparsity: Union[float, Dict[float, List[str]]],
        inter_func: str = "cubic",
        start_epoch: float = -1.0,
        min_start: float = -1.0,
        end_epoch: float = -1.0,
        min_end: float = -1.0,
        end_comparator: Union[int, None] = 0,
        update_frequency: float = -1.0,
        min_frequency: float = -1.0,
        global_sparsity: bool = False,
        allow_reintroduction: bool = False,
        parent_class_kwarg_names: Optional[List[str]] = None,
        **kwargs,
    ):
        self._final_sparsity_orig = final_sparsity
        self._params_orig = params
        params, self._final_sparsity = self._get_params_and_final_sparsity(
            params, final_sparsity
        )
        self._init_sparsity_orig = init_sparsity
        self._init_sparsity = init_sparsity
        self._inter_func = inter_func

        super().__init__(
            params=params,
            start_epoch=start_epoch,
            min_start=min_start,
            end_epoch=end_epoch,
            min_end=min_end,
            end_comparator=end_comparator,
            update_frequency=update_frequency,
            min_frequency=min_frequency,
            global_sparsity=global_sparsity,
            allow_reintroduction=allow_reintroduction,
            init_sparsity=self._init_sparsity,
            final_sparsity=self._final_sparsity,
            inter_func=inter_func,
            parent_class_kwarg_names=parent_class_kwarg_names,
            **kwargs,
        )

    def get_applied_sparsity_for_epoch(
        self, epoch: float, steps_per_epoch: int
    ) -> Union[float, List[float]]:
        """
        :param epoch: current epoch
        :param steps_per_epoch: number of steps in each epoch
        :return: sparsity level that should be applied based on the given interpolation
            function
        """
        init_sparsities = (
            self._init_sparsity
            if isinstance(self._init_sparsity, List)
            else [self._init_sparsity] * len(self.module_masks.layers)
        )
        final_sparsities = (
            self._final_sparsity
            if isinstance(self._final_sparsity, List)
            else [self._final_sparsity] * len(self.module_masks.layers)
        )
        return [
            interpolate(
                epoch,
                self.start_epoch,
                self.end_epoch,
                init_sparsity,
                final_sparsity,
                self._inter_func,
            )
            for init_sparsity, final_sparsity in zip(init_sparsities, final_sparsities)
        ]

    @ModifierProp()
    def init_sparsity(self) -> Union[float, str]:
        """
        :return: initial sparsity for each param to start with at
            start_epoch. __FROM_PARAM__ will set initial sparsity for each param to
            the existing sparsity level in that param
        """
        return self._init_sparsity_orig

    @init_sparsity.setter
    def init_sparsity(self, value: Union[float, str]):
        """
        :param value: initial sparsity for each param to start with at
            start_epoch. __FROM_PARAM__ will set initial sparsity for each param to
            the existing sparsity level in that param
        """
        if self.initialized:
            raise RuntimeError(
                f"init_sparsity may not be set after {self.__class__.__name__} has "
                "been initialized"
            )
        self._init_sparsity_orig = value
        self._init_sparsity = value

    @ModifierProp()
    def params(self) -> Union[str, List[str], None]:
        """
        :return: A list of full parameter names or regex patterns of names to apply
            pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
            will match to all parameters. __ALL_PRUNABLE__ will match to all ConvNd
            and Linear layers' weights
        """
        return self._params_orig

    @params.setter
    def params(self, value: Union[str, List[str], None]):
        """
        :params value: A list of full parameter names or regex patterns of names to
            apply pruning to.
            Regex patterns must be specified with the prefix 're:'. __ALL__
            will match to all parameters. __ALL_PRUNABLE__ will match to all ConvNd
            and Linear layers' weights
        """
        self._params_orig = value
        params, self._final_sparsity = self._get_params_and_final_sparsity(
            self._params_orig, self._final_sparsity_orig
        )
        self._params = validate_str_iterable(
            params, "{} for params".format(self.__class__.__name__)
        )

    @ModifierProp()
    def final_sparsity(self) -> Union[float, Dict[float, List[str]]]:
        """
        :return: the final sparsity for the param to end with at end_epoch
        """
        return self._final_sparsity_orig

    @final_sparsity.setter
    def final_sparsity(self, value: Union[float, Dict[float, List[str]]]):
        """
        :param value: the final sparsity for the param to end with at end_epoch.
            Can also be a Dict of final sparsity values to a list of parameters to apply
            them to. If given a Dict, then params must be set to [] and the params to
            be pruned will be read from the final_sparsity Dict
        """
        self._final_sparsity_orig = value
        self._params, self._final_sparsity = self._get_params_and_final_sparsity(
            self._params_orig, value
        )

    @ModifierProp()
    def inter_func(self) -> str:
        """
        :return: The type of interpolation function to use:
            [linear, cubic, inverse_cubic]
        """
        return self._inter_func

    def initialize_extras(self, module: Module):
        """
        Perform any extra setup for pruning modifier after params have been parsed,
        and mask creator and scorers initialized

        :param module: module to be pruned
        """
        super().initialize_extras(module)
        if self._init_sparsity == FROM_PARAM_TOKEN:
            self._init_sparsity = [
                tensor_sparsity(param).item() for param in self.module_masks.params_data
            ]

    def _create_named_layers_and_params(self, module: Module) -> List[NamedLayerParam]:
        if isinstance(self._final_sparsity_orig, float):
            return super()._create_named_layers_and_params(module)

        # update NamedLayerParam values to account for final sparsities dict

        final_sparsities = []
        named_layers_and_params = []
        added_names = set()

        for sparsity, param_names in self._final_sparsity_orig.items():
            layer_param_name_results = get_named_layers_and_params_by_regex(
                module,
                param_names,
                params_strict=True,
            )
            for result in layer_param_name_results:
                name = f"{result.layer_name}.{result.param_name}"
                if name not in added_names:
                    final_sparsities.append(sparsity)
                    named_layers_and_params.append(result)
                    added_names.add(name)
        self._final_sparsity = final_sparsities
        return named_layers_and_params

    @staticmethod
    def _get_params_and_final_sparsity(
        params: Union[str, List[str]],
        final_sparsity: Union[float, Dict[float, List[str]]],
    ) -> Tuple[Union[str, List[str]], Union[float, List[float]]]:
        if isinstance(final_sparsity, Dict):
            if params:
                raise ValueError(
                    "when final_sparsity is set to a Dict, params must be set to "
                    f"[]. Given final_sparsity: {final_sparsity} with params: "
                    f"{params}"
                )
            params = []
            sparsities = []
            for sparsity, sparsity_params in final_sparsity.items():
                for param in sparsity_params:
                    params.append(param)
                    sparsities.append(sparsity)
            return params, sparsities
        else:
            # default params to ALL_PRUNABLE_TOKEN
            params = params or ALL_PRUNABLE_TOKEN
            return params, final_sparsity
