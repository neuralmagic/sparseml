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
Modifiers for inducing / enforcing kernel sparsity (model pruning)
on models while pruning.
"""
import math
from abc import abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from sparseml.pytorch.nn import Identity
from sparseml.pytorch.optim.analyzer_pruning import ModulePruningAnalyzer
from sparseml.pytorch.optim.mask_creator_pruning import (
    PruningMaskCreator,
    load_mask_creator,
)
from sparseml.pytorch.optim.mask_pruning import ModuleParamPruningMask
from sparseml.pytorch.optim.modifier import (
    ModifierProp,
    PyTorchModifierYAML,
    ScheduledModifier,
    ScheduledUpdateModifier,
)
from sparseml.pytorch.utils import (
    GradSampler,
    MFACOptions,
    NamedLayerParam,
    get_layer,
    get_named_layers_and_params_by_regex,
    get_prunable_layers,
    replace_layer,
    tensor_sparsity,
)
from sparseml.pytorch.utils.logger import BaseLogger
from sparseml.utils import (
    ALL_PRUNABLE_TOKEN,
    ALL_TOKEN,
    INTERPOLATION_FUNCS,
    convert_to_bool,
    interpolate,
    validate_str_iterable,
)


__all__ = [
    "ConstantPruningModifier",
    "GMPruningModifier",
    "MagnitudePruningModifier",
    "MFACPruningModifier",
    "MFACGlobalPruningModifier",
    "MovementPruningModifier",
    "GlobalMagnitudePruningModifier",
    "LayerPruningModifier",
]


def _log_sparsity(
    tag_prefix: str,
    layer_sparsities: List[Union[Tuple[str, float], ModulePruningAnalyzer]],
    loggers: List[BaseLogger],
    epoch: float,
    steps_per_epoch: int,
):
    step = round(epoch) if steps_per_epoch <= 0 else round(epoch * steps_per_epoch)

    for logger in loggers:
        for layer_sparsity in layer_sparsities:
            if isinstance(layer_sparsity, ModulePruningAnalyzer):
                layer_sparsity = (
                    layer_sparsity.tag,
                    layer_sparsity.param_sparsity.item(),
                )

            logger.log_scalar(
                f"{tag_prefix}/{layer_sparsity[0]}",
                layer_sparsity[1],
                step,
            )


class _PruningParamsModifier(ScheduledUpdateModifier):
    """
    Base class for pruning modifiers that create masks for given params

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
        if == -1, then end_epoch can be less than, equal, or greater than start_epoch.
        if == 0, then end_epoch can be equal to or greater than start_epoch.
        if == 1, then end_epoch can only be greater than start_epoch.
    :param update_frequency: The number of epochs or fraction of epochs to
            update at between start and end
    :param min_frequency: The minimum acceptable value for update_frequency,
        default -1
    :param log_types: The loggers to allow the learning rate to be logged to,
        default is __ALL__
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
        log_types: Union[str, List[str]] = None,
    ):
        super().__init__(
            log_types=log_types,
            start_epoch=start_epoch,
            min_start=min_start,
            end_epoch=end_epoch,
            min_end=min_end,
            end_comparator=end_comparator,
            update_frequency=update_frequency,
            min_frequency=min_frequency,
        )
        self._params = validate_str_iterable(
            params, "{} for params".format(self.__class__.__name__)
        )
        self._module_masks = None  # type: Optional[ModuleParamPruningMask]
        self._analyzers = None  # type: Optional[List[ModulePruningAnalyzer]]
        self._last_logged_epoch = None

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
        if diff:
            raise IndexError(
                f"Found extra keys: {state_dict_keys - mask_names} "
                f"and missing keys: {mask_names - state_dict_keys}"
            )

        self._module_masks.set_param_masks(
            [state_dict[name] for name in self._module_masks.names]
        )

    @ModifierProp()
    def params(self) -> Union[str, List[str]]:
        """
        :return: A list of full parameter names or regex patterns of names to apply
            pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
            will match to all parameters. __ALL_PRUNABLE__ will match to all ConvNd
            and Linear layers' weights
        """
        return self._params

    @params.setter
    def params(self, value: Union[str, List[str]]):
        """
        :params value: A list of full parameter names or regex patterns of names to
            apply pruning to.
            Regex patterns must be specified with the prefix 're:'. __ALL__
            will match to all parameters. __ALL_PRUNABLE__ will match to all ConvNd
            and Linear layers' weights
        """
        self._params = validate_str_iterable(
            value, "{} for params".format(self.__class__.__name__)
        )

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

    def initialize(
        self,
        module: Module,
        epoch: float = 0,
        loggers: Optional[List[BaseLogger]] = None,
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
        self._module_masks = self._create_pruning_mask(layers, layer_names, param_names)
        self._analyzers = self._create_analyzers(layers, layer_names, param_names)

        if len(self._analyzers) == 0:
            raise ValueError(
                "Could not find any params matching {} in {}".format(
                    self._params, self.__class__.__name__
                )
            )

        self._check_mask_update(module, epoch, steps_per_epoch=1, **kwargs)

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
        self._check_mask_update(module, epoch, steps_per_epoch)

    def log_update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
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

        if self._should_log(module, optimizer, epoch, steps_per_epoch):
            self._last_logged_epoch = math.floor(epoch)
            _log_sparsity(
                "ParamPruning", self._analyzers, self.loggers, epoch, steps_per_epoch
            )

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

        # be sure to apply mask again after optimizer update because
        # weights may have changed (optimizer with momentum, not masking gradient)
        self._module_masks.apply()

    @abstractmethod
    def _check_mask_update(
        self, module: Module, epoch: float, steps_per_epoch: int, **kwargs
    ):
        raise NotImplementedError()

    def _should_log(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ) -> bool:
        return self._last_logged_epoch != math.floor(epoch)

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
        return ModuleParamPruningMask(layers, param_names, layer_names=layer_names)

    def _create_analyzers(
        self, layers: List[Module], layer_names: List[str], param_names: List[str]
    ):
        return [
            ModulePruningAnalyzer(layer, layer_name, param_name)
            for (layer, layer_name, param_name) in zip(layers, layer_names, param_names)
        ]


@PyTorchModifierYAML()
class ConstantPruningModifier(_PruningParamsModifier):
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
        super().__init__(
            params=params,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            end_comparator=-1,
            update_frequency=-1,
            log_types=log_types,
        )

    def _check_mask_update(
        self, module: Module, epoch: float, steps_per_epoch: int, **kwargs
    ):
        if self.start_pending(epoch, steps_per_epoch):
            self._module_masks.set_param_masks_from_weights()
            self._module_masks.enabled = True

        if self.end_pending(epoch, steps_per_epoch):
            self._module_masks.set_param_masks_from_weights()
            self._module_masks.enabled = False


@PyTorchModifierYAML()
class GMPruningModifier(_PruningParamsModifier):
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
    |       log_types: __ALL__
    |       mask_type: unstructured
    |       global_sparsity: False
    |       score_type: magnitude

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
    :param phased: True to enable a phased approach where pruning will
        turn on and off with the update_frequency. Starts with pruning on
        at start_epoch, off at start_epoch + update_frequency, and so on.
    :param log_types: The loggers to allow the learning rate to be logged to,
        default is __ALL__
    :param mask_type: String to define type of sparsity (options: ['unstructured',
        'channel', 'filter']), List to define block shape of a parameters in and out
        channels, or a SparsityMaskCreator object. default is 'unstructured'
    :param global_sparsity: set True to enable global pruning. if False, pruning will
        be layer-wise. Default is False
    :param score_type: Method used to score parameters for masking, i.e.
        'magnitude', 'movement'. Default is 'magnitude'
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
        phased: bool = False,
        log_types: Union[str, List[str]] = ALL_TOKEN,
        mask_type: Union[str, List[int], PruningMaskCreator] = "unstructured",
        global_sparsity: bool = False,
        score_type: Union[str, MFACOptions] = "magnitude",
    ):
        self._final_sparsity_orig = final_sparsity
        self._params_orig = params
        params, self._final_sparsity = self._get_params_and_final_sparsity(
            params, final_sparsity
        )
        super().__init__(
            params=params,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            end_comparator=-1,
            update_frequency=update_frequency,
            log_types=log_types,
        )
        self._init_sparsity = init_sparsity
        self._leave_enabled = convert_to_bool(leave_enabled)
        self._inter_func = inter_func
        self._phased = phased
        self._mask_type = mask_type
        self._mask_creator = (
            mask_type
            if isinstance(mask_type, PruningMaskCreator)
            else load_mask_creator(mask_type)
        )
        self._global_sparsity = global_sparsity
        self._score_type = score_type
        self._applied_sparsity = None
        self._last_logged_sparsity = None
        self._pre_step_completed = False

        self._non_serializable_props = {}

        self.validate()

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
    def init_sparsity(self) -> float:
        """
        :return: the initial sparsity for the param to start with at start_epoch
        """
        return self._init_sparsity

    @init_sparsity.setter
    def init_sparsity(self, value: float):
        """
        :param value: the initial sparsity for the param to start with at start_epoch
        """
        self._init_sparsity = value
        self.validate()

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
        self.validate()

    @ModifierProp()
    def leave_enabled(self) -> bool:
        """
        :return: True to continue masking the weights after end_epoch,
            False to stop masking. Note, if set as False, sparsity will not be enforced
            and the model will likely deviate from the sparse solution
        """
        return self._leave_enabled

    @leave_enabled.setter
    def leave_enabled(self, value: bool):
        """
        :param value: True to continue masking the weights after end_epoch,
            False to stop masking. Note, if set as False, sparsity will not be enforced
            and the model will likely deviate from the sparse solution
        """
        self._leave_enabled = value

    @ModifierProp()
    def inter_func(self) -> str:
        """
        :return: the type of interpolation function to use:
            [linear, cubic, inverse_cubic]
        """
        return self._inter_func

    @inter_func.setter
    def inter_func(self, value: str):
        """
        :param value: the type of interpolation function to use:
            [linear, cubic, inverse_cubic]
        """
        self._inter_func = value
        self.validate()

    @ModifierProp()
    def phased(self) -> bool:
        """
        :return: True to enable a phased approach where pruning will
            turn on and off with the update_frequency. Starts with pruning on
            at start_epoch, off at start_epoch + update_frequency, and so on.
        """
        return self._phased

    @phased.setter
    def phased(self, value: bool):
        """
        :param value: the type of interpolation function to use:
            [linear, cubic, inverse_cubic]
        """
        self._phased = value
        self.validate()

    @ModifierProp()
    def mask_type(self) -> Union[str, List[int], PruningMaskCreator]:
        """
        :return: the SparsityMaskCreator object used
        """
        return self._mask_type

    @mask_type.setter
    def mask_type(self, value: Union[str, List[int], PruningMaskCreator]):
        """
        :param value: the SparsityMaskCreator object to use
        """
        self._mask_type = value
        self._mask_creator = value
        if not isinstance(value, PruningMaskCreator):
            self._mask_creator = load_mask_creator(value)

    @ModifierProp()
    def global_sparsity(self) -> bool:
        """
        :return: True if global pruning is enabled, False otherwise
        """
        return self._global_sparsity

    @ModifierProp()
    def score_type(self) -> Union[str, MFACOptions]:
        """
        :return: the the scoring method used for pruning
        """
        return self._score_type

    @ModifierProp(serializable=False)
    def applied_sparsity(self) -> Union[float, List[float]]:
        """
        :return: the currently applied sparsity level to each of the contained params
        """
        return self._applied_sparsity

    def optimizer_pre_step(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Update mask movement scores with gradients right before optimizer step is
        applied. Called here in case gradients are changed between the backwards
        pass and step such as in grad norm clipping
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
        has momentum that may have moved weights from 0. Not applied for
        movement pruning to allow weight reintroduction

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super(_PruningParamsModifier, self).optimizer_post_step(
            module, optimizer, epoch, steps_per_epoch
        )

        if not self._module_masks.allow_reintroduction:
            # be sure to apply mask again after optimizer update because weights may
            # have changed (optimizer with momentum, not masking gradient)
            self._module_masks.apply()

    def validate(self):
        """
        Validate the values of the params for the current instance are valid
        """

        if not isinstance(self._init_sparsity, float):
            raise TypeError(
                "init_sparsity must be of float type for {}".format(
                    self.__class__.__name__
                )
            )

        if self._init_sparsity < 0.0 or self._init_sparsity > 1.0:
            raise ValueError(
                (
                    "init_sparsity value must be in the range [0.0, 1.0],"
                    " given {} for {}"
                ).format(self._init_sparsity, self.__class__.__name__)
            )

        if not isinstance(self._final_sparsity_orig, float) and not isinstance(
            self._final_sparsity_orig, Dict
        ):
            raise TypeError(
                "final_sparsity must be of float or Dict type for {}".format(
                    self.__class__.__name__
                )
            )

        final_sparsities = (
            self._final_sparsity
            if isinstance(self._final_sparsity, List)
            else [self._final_sparsity]
        )
        if not all(0.0 < sparsity < 1.0 for sparsity in final_sparsities):
            raise ValueError(
                (
                    "final_sparsity value(s) must be in the range (0.0, 1.0),"
                    " given {} for {}"
                ).format(self._final_sparsity, self.__class__.__name__)
            )

        if self._inter_func not in INTERPOLATION_FUNCS:
            raise ValueError(
                (
                    "{} is not a supported inter_func in layers_settings,"
                    " available are {} for {}"
                ).format(self._inter_func, INTERPOLATION_FUNCS, self.__class__.__name__)
            )

    def _check_mask_update(
        self, module: Module, epoch: float, steps_per_epoch: int, **kwargs
    ):
        """
        Check for updating the pruning masks at the given epoch.
        Called from both initialize and update.

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
            self._module_masks.pre_optim_step_update()
            self._pre_step_completed = True

        if started:
            # set the mask tensors according to the new sparsity
            if isinstance(self._final_sparsity, List):
                self._applied_sparsity = [
                    interpolate(
                        epoch,
                        self.start_epoch,
                        self.end_epoch,
                        self._init_sparsity,
                        final_sparsity,
                        self._inter_func,
                    )
                    for final_sparsity in self._final_sparsity
                ]
            else:
                self._applied_sparsity = interpolate(
                    epoch,
                    self.start_epoch,
                    self.end_epoch,
                    self._init_sparsity,
                    self._final_sparsity,
                    self._inter_func,
                )

            # make sure if phased that the phases end at the final sparsity
            # if it doesn't divide evenly
            if self.phased and not self.end_pending(epoch, steps_per_epoch):
                # adjust for phased pruning: start=on, start+update=off
                phase = math.floor((epoch - self.start_epoch) / self.update_frequency)
                if phase % 2 != 0:
                    # odd update phase, turn sparsity off
                    self._applied_sparsity = (
                        0.0
                        if not isinstance(self._applied_sparsity, List)
                        else [0.0] * len(self._applied_sparsity)
                    )

            self._module_masks.set_param_masks_from_sparsity(self._applied_sparsity)

        if self.end_pending(epoch, steps_per_epoch):
            self._module_masks.pruning_end(self._leave_enabled)

    def _should_log(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ) -> bool:
        """
        Log on applied sparsity change or epoch change.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        :return: True if the sparsity should be logged for the pruning modifier,
            False otherwise.
        """
        if (
            self._applied_sparsity != self._last_logged_sparsity
            or math.floor(epoch) != self._last_logged_epoch
        ):
            self._last_logged_sparsity = self._applied_sparsity
            return True

        return False

    def _create_pruning_mask(
        self, layers: List[Module], layer_names: List[str], param_names: List[str]
    ) -> ModuleParamPruningMask:
        return ModuleParamPruningMask(
            layers,
            param_names,
            layer_names=layer_names,
            mask_creator=self._mask_creator,
            global_sparsity=self._global_sparsity,
            score_type=self._score_type,
        )

    def _create_named_layers_and_params(self, module: Module) -> List[NamedLayerParam]:
        if isinstance(self._final_sparsity_orig, float):
            return super()._create_named_layers_and_params(module)

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


@PyTorchModifierYAML()
class MagnitudePruningModifier(GMPruningModifier):
    """
    Gradually applies kernel sparsity to a given parameter or parameters from
    init_sparsity until final_sparsity is reached over a given amount of time
    and applied with an interpolated function for each step taken.

    Uses magnitude pruning to gradually mask parameter values. Pruning is
    unstructured by default, structure can be specified by mask_type.

    | Sample yaml:
    |   !MagnitudePruningModifier
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
    :param phased: True to enable a phased approach where pruning will
        turn on and off with the update_frequency. Starts with pruning on
        at start_epoch, off at start_epoch + update_frequency, and so on.
    :param log_types: The loggers to allow the learning rate to be logged to,
        default is __ALL__
    :param mask_type: String to define type of sparsity (options: ['unstructured',
        'channel', 'filter']), List to define block shape of a parameters in and out
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
        phased: bool = False,
        log_types: Union[str, List[str]] = ALL_TOKEN,
        mask_type: Union[str, List[int], PruningMaskCreator] = "unstructured",
    ):
        super().__init__(
            init_sparsity=init_sparsity,
            final_sparsity=final_sparsity,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            params=params,
            leave_enabled=leave_enabled,
            inter_func=inter_func,
            phased=phased,
            log_types=log_types,
            mask_type=mask_type,
            global_sparsity=False,
            score_type="magnitude",
        )

    @ModifierProp(serializable=False)
    def global_sparsity(self) -> bool:
        """
        :return: True if global pruning is enabled, False otherwise
        """
        return self._global_sparsity

    @ModifierProp(serializable=False)
    def score_type(self) -> str:
        """
        :return: the the scoring method used for pruning
        """
        return self._score_type


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
    |       log_types: __ALL__
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
    :param phased: True to enable a phased approach where pruning will
        turn on and off with the update_frequency. Starts with pruning on
        at start_epoch, off at start_epoch + update_frequency, and so on.
    :param log_types: The loggers to allow the learning rate to be logged to,
        default is __ALL__
    :param mask_type: String to define type of sparsity (options: ['unstructured',
        'channel', 'filter']), List to define block shape of a parameters in and out
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
        phased: bool = False,
        log_types: Union[str, List[str]] = ALL_TOKEN,
        mask_type: Union[str, List[int], PruningMaskCreator] = "unstructured",
    ):
        super().__init__(
            init_sparsity=init_sparsity,
            final_sparsity=final_sparsity,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            params=params,
            leave_enabled=leave_enabled,
            inter_func=inter_func,
            phased=phased,
            log_types=log_types,
            mask_type=mask_type,
            global_sparsity=False,
            score_type="movement",
        )

    @ModifierProp(serializable=False)
    def global_sparsity(self) -> bool:
        """
        :return: True if global pruning is enabled, False otherwise
        """
        return self._global_sparsity

    @ModifierProp(serializable=False)
    def score_type(self) -> str:
        """
        :return: the the scoring method used for pruning
        """
        return self._score_type


@PyTorchModifierYAML()
class GlobalMagnitudePruningModifier(GMPruningModifier):
    """
    Gradually applies kernel sparsity to a given parameter or parameters from
    init_sparsity until final_sparsity is reached over a given amount of time
    and applied with an interpolated function for each step taken.

    Uses magnitude pruning over the global scope of all given parameters
    to gradually mask parameter values. Pruning is unstructured by default,
    structure can be specified by mask_type.

    | Sample yaml:
    |   !GlobalMagnitudePruningModifier
    |       init_sparsity: 0.05
    |       final_sparsity: 0.8
    |       start_epoch: 0.0
    |       end_epoch: 10.0
    |       update_frequency: 1.0
    |       params: __ALL_PRUNABLE__
    |       leave_enabled: True
    |       inter_func: cubic
    |       log_types: __ALL__
    |       mask_type: unstructured
    |       score_type: magnitude

    :param init_sparsity: the initial sparsity for the param to start with at
        start_epoch
    :param final_sparsity: the final sparsity for the param to end with at end_epoch
    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param update_frequency: The number of epochs or fraction of epochs to update at
        between start and end
    :param params: A list of full parameter names or regex patterns of names to apply
        pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
        will match to all parameters. __ALL_PRUNABLE__ will match to all ConvNd
        and Linear layers' weights. Defualt is __ALL_PRUNABLE__
    :param leave_enabled: True to continue masking the weights after end_epoch,
        False to stop masking. Should be set to False if exporting the result
        immediately after or doing some other prune
    :param inter_func: the type of interpolation function to use:
        [linear, cubic, inverse_cubic]
    :param phased: True to enable a phased approach where pruning will
        turn on and off with the update_frequency. Starts with pruning on
        at start_epoch, off at start_epoch + update_frequency, and so on.
    :param log_types: The loggers to allow the learning rate to be logged to,
        default is __ALL__
    :param mask_type: String to define type of sparsity (options: ['unstructured',
        'channel', 'filter']), List to define block shape of a parameters in and out
        channels, or a SparsityMaskCreator object. default is 'unstructured'
    :param score_type: Method used to score parameters for masking, i.e.
        'magnitude', 'movement'. Default is 'magnitude'
    """

    def __init__(
        self,
        init_sparsity: float,
        final_sparsity: float,
        start_epoch: float,
        end_epoch: float,
        update_frequency: float,
        params: Union[str, List[str]] = ALL_PRUNABLE_TOKEN,
        leave_enabled: bool = True,
        inter_func: str = "cubic",
        phased: bool = False,
        log_types: Union[str, List[str]] = ALL_TOKEN,
        mask_type: Union[str, List[int], PruningMaskCreator] = "unstructured",
        score_type: Union[str, MFACOptions] = "magnitude",
    ):
        super().__init__(
            init_sparsity=init_sparsity,
            final_sparsity=final_sparsity,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            params=params,
            leave_enabled=leave_enabled,
            inter_func=inter_func,
            phased=phased,
            log_types=log_types,
            mask_type=mask_type,
            global_sparsity=True,
            score_type=score_type,
        )

    @ModifierProp(serializable=False)
    def global_sparsity(self) -> bool:
        """
        :return: True if global pruning is enabled, False otherwise
        """
        return self._global_sparsity


@PyTorchModifierYAML()
class MFACPruningModifier(GMPruningModifier):
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
    :param phased: True to enable a phased approach where pruning will
        turn on and off with the update_frequency. Starts with pruning on
        at start_epoch, off at start_epoch + update_frequency, and so on.
    :param log_types: The loggers to allow the learning rate to be logged to,
        default is __ALL__
    :param mask_type: String to define type of sparsity (options: ['unstructured',
        'channel', 'filter']), List to define block shape of a parameters in and out
        channels, or a SparsityMaskCreator object. default is 'unstructured'
    :param global_sparsity: set True to enable global pruning. if False, pruning will
        be layer-wise. Default is False
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
        phased: bool = False,
        log_types: Union[str, List[str]] = ALL_TOKEN,
        mask_type: Union[str, List[int], PruningMaskCreator] = "unstructured",
        global_sparsity: bool = False,
        mfac_options: Dict[str, Any] = None,
    ):
        super().__init__(
            init_sparsity=init_sparsity,
            final_sparsity=final_sparsity,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            params=params,
            leave_enabled=leave_enabled,
            inter_func=inter_func,
            phased=phased,
            log_types=log_types,
            mask_type=mask_type,
            global_sparsity=global_sparsity,
            score_type="MFAC",
        )
        self._mfac_options = mfac_options or {}

    @ModifierProp(serializable=False)
    def score_type(self) -> str:
        """
        :return: the the scoring method used for pruning
        """
        return self._score_type

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

    def _check_mask_update(
        self, module: Module, epoch: float, steps_per_epoch: int, **kwargs
    ):
        # create grads for pne-shot pruning
        if "grad_sampler" in kwargs:
            self._collect_grad_samples(module, kwargs["grad_sampler"])
            self._pre_step_completed = True

        super()._check_mask_update(module, epoch, steps_per_epoch, **kwargs)

    def _create_pruning_mask(
        self, layers: List[Module], layer_names: List[str], param_names: List[str]
    ) -> ModuleParamPruningMask:
        return ModuleParamPruningMask(
            layers,
            param_names,
            layer_names=layer_names,
            mask_creator=self._mask_creator,
            global_sparsity=self._global_sparsity,
            score_type=MFACOptions(**self._mfac_options),
        )

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

        for _ in grad_sampler.iter_module_backwards(module, num_grads):
            self._module_masks.pre_optim_step_update()


@PyTorchModifierYAML()
class MFACGlobalPruningModifier(MFACPruningModifier):
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
    :param phased: True to enable a phased approach where pruning will
        turn on and off with the update_frequency. Starts with pruning on
        at start_epoch, off at start_epoch + update_frequency, and so on.
    :param log_types: The loggers to allow the learning rate to be logged to,
        default is __ALL__
    :param mask_type: String to define type of sparsity (options: ['unstructured',
        'channel', 'filter']), List to define block shape of a parameters in and out
        channels, or a SparsityMaskCreator object. default is 'unstructured'
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
        phased: bool = False,
        log_types: Union[str, List[str]] = ALL_TOKEN,
        mask_type: Union[str, List[int], PruningMaskCreator] = "unstructured",
        mfac_options: Dict[str, Any] = None,
    ):
        super().__init__(
            init_sparsity=init_sparsity,
            final_sparsity=final_sparsity,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            params=params,
            leave_enabled=leave_enabled,
            inter_func=inter_func,
            phased=phased,
            log_types=log_types,
            mask_type=mask_type,
            global_sparsity=True,
            mfac_options=mfac_options,
        )
        self._mfac_options = mfac_options or {}

    @ModifierProp(serializable=False)
    def global_sparsity(self) -> bool:
        """
        :return: True if global pruning is enabled, False otherwise
        """
        return self._global_sparsity


@PyTorchModifierYAML()
class LayerPruningModifier(ScheduledUpdateModifier):
    """
    Class for pruning away layers within a module
    (replaces with sparseml.pytorch.nn.Identity).

    | Sample yaml:
    |   !LayerPruningModifier
    |       layers: ['bert.encoder.layer.6', 'bert.encoder.layer.7']
    |

    :param layers: A list of full layer names to apply pruning to.
        __ALL_ will match to all layers. __ALL_PRUNABLE__ will match to all ConvNd
        and Linear layers
    :param start_epoch: The epoch the modifier will prune layers away layers at
    :param end_epoch: The epoch, if set and positive,
        the modifier will reintroduce the pruned layers at
    :param update_frequency: Unused for this modifier
    :param log_types: The loggers to allow the learning rate to be logged to,
        default is __ALL__
    """

    def __init__(
        self,
        layers: Union[str, List[str]],
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
        update_frequency: float = -1.0,
        log_types: Union[str, List[str]] = None,
    ):
        super().__init__(
            log_types=log_types,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=-1.0,
        )
        self._layers = validate_str_iterable(
            layers, "{} for layers".format(self.__class__.__name__)
        )
        self._layer_modules = {}  # type: Dict[str, Module]
        self._layers_replaced = False
        self._last_logged_epoch = None
        self._last_logged_layers_replaced = None

    @ModifierProp()
    def layers(self) -> Union[str, List[str]]:
        """
        :return: the layers to prune from the module
        """
        return self._layers

    @layers.setter
    def layers(self, value: Union[str, List[str]]):
        """
        :param value: the layers to prune from the module
        """
        self._layers = validate_str_iterable(
            value, "{} for layers".format(self.__class__.__name__)
        )

    def initialize(
        self,
        module: Module,
        epoch: float = 0,
        loggers: Optional[List[BaseLogger]] = None,
        **kwargs,
    ):
        """
        Grab the layers and apply if epoch in range to control pruning for.

        :param module: the PyTorch model/module to modify
        :param epoch: The epoch to initialize the modifier and module at.
            Defaults to 0 (start of the training process)
        :param loggers: Optional list of loggers to log the modification process to
        :param kwargs: Optional kwargs to support specific arguments
            for individual modifiers.
        """
        super().initialize(module, epoch, loggers, **kwargs)
        layers = self._get_named_layers(module)

        for (layer_name, layer_module) in layers:
            self._layer_modules[layer_name] = None

        self._check_update_pruning(module, epoch, steps_per_epoch=1)

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
        self._check_update_pruning(module, epoch=math.inf, steps_per_epoch=1)
        self._layer_modules = None
        self._last_logged_epoch = None
        self._last_logged_layers_replaced = None

    def update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Update to enable and disable the layers when chosen.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().update(module, optimizer, epoch, steps_per_epoch)
        self._check_update_pruning(module, epoch, steps_per_epoch)

    def log_update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Check whether to log an update for the state of the modifier.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().log_update(module, optimizer, epoch, steps_per_epoch)

        if self._should_log(module, optimizer, epoch, steps_per_epoch):
            self._last_logged_epoch = math.floor(epoch)
            self._last_logged_layers_replaced = self._layers_replaced
            _log_sparsity(
                "LayerPruning",
                [
                    (name, 1 if self._layers_replaced else 0)
                    for name in self._layer_modules.keys()
                ],
                self.loggers,
                epoch,
                steps_per_epoch,
            )

    def _check_layers_match(self, token: Union[str, List[str]]):
        if isinstance(token, str):
            return token in self._layers or token == self._layers

        if isinstance(self._layers, str):
            return self._layers in token

        return len(set(token).intersection(set(self._layers))) > 0

    def _get_named_layers(self, module: Module) -> List[Tuple[str, Module]]:
        if self._check_layers_match(ALL_TOKEN):
            layers = [(name, layer) for name, layer in module.named_modules()]
        elif self._check_layers_match(ALL_PRUNABLE_TOKEN):
            layers = get_prunable_layers(module)
        else:
            layers = [(name, get_layer(name, module)) for name in self._layers]

        return layers

    def _check_update_pruning(self, module: Module, epoch: float, steps_per_epoch: int):
        if not self._layers_replaced and (
            epoch >= self.start_epoch or self.start_epoch == -1
        ):
            for name in list(self._layer_modules.keys()):
                self._layer_modules[name] = replace_layer(module, name, Identity())
            self._layers_replaced = True

        if self._layers_replaced and (epoch >= self.end_epoch and self.end_epoch != -1):
            for name, replaced in self._layer_modules.items():
                replace_layer(module, name, replaced)
                self._layer_modules[name] = None
            self._layers_replaced = False

    def _should_log(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ) -> bool:
        return (
            self._last_logged_epoch != math.floor(epoch)
            or self._last_logged_layers_replaced != self._layers_replaced
        )
