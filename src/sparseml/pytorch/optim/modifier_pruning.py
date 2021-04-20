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
from collections import OrderedDict
from typing import Dict, List, Union

from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer

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
    get_named_layers_and_params_by_regex,
    get_prunable_layers,
    tensor_sparsity,
)
from sparseml.pytorch.utils.logger import PyTorchLogger
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
    "GlobalMagnitudePruningModifier",
]


def _log_sparsity(
    analyzers: List[ModulePruningAnalyzer],
    loggers: List[PyTorchLogger],
    epoch: float,
    steps_per_epoch: int,
):
    step = round(epoch) if steps_per_epoch <= 0 else round(epoch * steps_per_epoch)

    for logger in loggers:
        for analyzer in analyzers:
            logger.log_scalar(
                "Modifier KS/{}".format(analyzer.tag),
                analyzer.param_sparsity.item(),
                step,
            )


@PyTorchModifierYAML()
class ConstantPruningModifier(ScheduledModifier):
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

            if sparsity > 0.4:
                modifiers.append(
                    ConstantPruningModifier(params=["{}.{}".format(name, "weight")])
                )

        return modifiers

    def __init__(
        self,
        params: Union[str, List[str]],
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
        log_types: Union[str, List[str]] = ALL_TOKEN,
    ):
        super().__init__(
            log_types=log_types,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            end_comparator=-1,
        )
        self._params = validate_str_iterable(
            params, "{} for params".format(self.__class__.__name__)
        )
        self._module_masks = None  # type: ModuleParamPruningMask
        self._analyzers = None
        self._last_logged_epoch = None

    def __del__(self):
        del self._module_masks

    def state_dict(self) -> Dict[str, Tensor]:
        """
        :return: Dictionary to store the masks currently created by this object. The
            mapping is param_name -> mask
        """
        return OrderedDict(
            zip(self._module_masks.names, self._module_masks.param_masks)
        )

    def load_state_dict(self, state_dict: Dict[str, Tensor]):
        """
        Loads the given state dict into this object's modifiers

        :param state_dict: dictionary object as generated by this object's state_dict
            function
        :raises RuntimeError: If any keys in the state dict do not correspond to a valid
            parameter in this modifier or if this modifier has not been initialized
        """
        if not self.initialized:
            raise RuntimeError("Cannot load state dict for an uninitialized modifier")

        mask_names = self._module_masks.names
        for key in state_dict:
            if key not in mask_names:
                raise RuntimeError(
                    f"State dict key {key} not found in ConstantPruningModifier params"
                )
        for name in mask_names:
            if name not in state_dict:
                raise RuntimeError(
                    f"Mask parameter name {name} not found in state dict"
                )

        masks_disabled = False
        if not masks_disabled:
            # enable mask object so that the laoded mask will apply
            masks_disabled = True
        self._module_masks.set_param_masks([state_dict[name] for name in mask_names])
        if masks_disabled:
            # set mask object back to disabled
            self._module_masks.enabled = True

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

    def initialize(self, module: Module, optimizer: Optimizer):
        """
        Grab the params to control kernel sparsity for.

        :param module: module to modify
        :param optimizer: optimizer to modify
        """
        super().initialize(module, optimizer)

        if self._params == ALL_TOKEN or ALL_TOKEN in self._params:
            param_names = ["re:.*"]
        elif self._params == ALL_PRUNABLE_TOKEN or ALL_PRUNABLE_TOKEN in self._params:
            param_names = [
                name + ".weight" for (name, _) in get_prunable_layers(module)
            ]
        else:
            param_names = self._params

        named_layers_and_params = get_named_layers_and_params_by_regex(
            module,
            param_names,
            params_strict=True,
        )

        layers = [
            named_layer_param.layer for named_layer_param in named_layers_and_params
        ]
        param_names = [
            named_layer_param.param_name
            for named_layer_param in named_layers_and_params
        ]
        layer_names = [
            named_layer_param.layer_name
            for named_layer_param in named_layers_and_params
        ]
        self._module_masks = ModuleParamPruningMask(
            layers, param_names, layer_names=layer_names
        )

        self._analyzers = []
        for layer_name, layer, param_name, _ in named_layers_and_params:
            self._analyzers.append(ModulePruningAnalyzer(layer, layer_name, param_name))

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

        if self.start_pending(epoch, steps_per_epoch):
            self._module_masks.set_param_masks_from_weights()
            self._module_masks.enabled = True

        if self.end_pending(epoch, steps_per_epoch):
            self._module_masks.set_param_masks_from_weights()
            self._module_masks.enabled = False

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

        if self._last_logged_epoch != math.floor(epoch):
            self._last_logged_epoch = math.floor(epoch)
            _log_sparsity(self._analyzers, self.loggers, epoch, steps_per_epoch)

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


@PyTorchModifierYAML()
class GMPruningModifier(ScheduledUpdateModifier):
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
        and Linear layers' weights
    :param leave_enabled: True to continue masking the weights after end_epoch,
        False to stop masking. Should be set to False if exporting the result
        immediately after or doing some other prune
    :param inter_func: the type of interpolation function to use:
        [linear, cubic, inverse_cubic]
    :param log_types: The loggers to allow the learning rate to be logged to,
        default is __ALL__
    :param mask_type: String to define type of sparsity (options: ['unstructured',
        'channel', 'filter']), List to define block shape of a parameters in and out
        channels, or a SparsityMaskCreator object. default is 'unstructured'
    :param global_sparsity: set True to enable global pruning. if False, pruning will
        be layer-wise. Default is False
    """

    def __init__(
        self,
        init_sparsity: float,
        final_sparsity: float,
        start_epoch: float,
        end_epoch: float,
        update_frequency: float,
        params: Union[str, List[str]],
        leave_enabled: bool = True,
        inter_func: str = "cubic",
        log_types: Union[str, List[str]] = ALL_TOKEN,
        mask_type: Union[str, List[int], PruningMaskCreator] = "unstructured",
        global_sparsity: bool = False,
    ):
        super().__init__(
            log_types=log_types,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            min_end=0.0,
            end_comparator=1,
        )
        self._init_sparsity = init_sparsity
        self._final_sparsity = final_sparsity
        self._params = validate_str_iterable(
            params, "{} for params".format(self.__class__.__name__)
        )
        self._leave_enabled = convert_to_bool(leave_enabled)
        self._inter_func = inter_func
        self._mask_type = mask_type
        self._mask_creator = mask_type
        if not isinstance(mask_type, PruningMaskCreator):
            self._mask_creator = load_mask_creator(mask_type)
        self._global_sparsity = global_sparsity
        self._module_masks = None  # type: ModuleParamPruningMask
        self._applied_sparsity = None
        self._last_logged_sparsity = None
        self._last_logged_epoch = None
        self._analyzers = None

        self._non_serializable_props = {}

        self.validate()

    def __del__(self):
        del self._module_masks

    def state_dict(self) -> Dict[str, Tensor]:
        """
        :return: Dictionary to store the masks currently created by this object. The
            mapping is param_name -> mask
        """
        return OrderedDict(
            zip(self._module_masks.names, self._module_masks.param_masks)
        )

    def load_state_dict(self, state_dict: Dict[str, Tensor]):
        """
        Loads the given state dict into this object's modifiers

        :param state_dict: dictionary object as generated by this object's state_dict
            function
        :raises RuntimeError: If any keys in the state dict do not correspond to a valid
            parameter in this modifier or if this modifier has not been initialized
        """
        if not self.initialized:
            raise RuntimeError("Cannot load state dict for an uninitialized modifier")

        mask_names = self._module_masks.names
        for key in state_dict:
            if key not in mask_names:
                raise RuntimeError(
                    f"State dict key {key} not found in ConstantPruningModifier params"
                )
        for name in mask_names:
            if name not in state_dict:
                raise RuntimeError(
                    f"Mask parameter name {name} not found in state dict"
                )

        masks_disabled = False
        if not masks_disabled:
            # enable mask object so that the laoded mask will apply
            masks_disabled = True
        self._module_masks.set_param_masks([state_dict[name] for name in mask_names])
        if masks_disabled:
            # set mask object back to disabled
            self._module_masks.enabled = True

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
    def final_sparsity(self) -> float:
        """
        :return: the final sparsity for the param to end with at end_epoch
        """
        return self._final_sparsity

    @final_sparsity.setter
    def final_sparsity(self, value: float):
        """
        :param value: the final sparsity for the param to end with at end_epoch
        """
        self._final_sparsity = value
        self.validate()

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
        :param value: A list of full parameter names or regex patterns of names to apply
            pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
            will match to all parameters. __ALL_PRUNABLE__ will match to all ConvNd
            and Linear layers' weights
        """
        self._params = validate_str_iterable(
            value, "{} for params".format(self.__class__.__name__)
        )

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

    def global_sparsity(self) -> bool:
        """
        :return: True if global pruning is enabled, False otherwise
        """
        return self._global_sparsity

    @ModifierProp(serializable=False)
    def applied_sparsity(self) -> float:
        """
        :return: the currently applied sparsity level to the contained params
        """
        return self._applied_sparsity

    def initialize(self, module: Module, optimizer: Optimizer):
        """
        Grab the params to control kernel sparsity for

        :param module: module to modify
        :param optimizer: optimizer to modify
        """
        super().initialize(module, optimizer)

        if self._params == ALL_TOKEN or ALL_TOKEN in self._params:
            param_names = ["re:.*"]
        elif self._params == ALL_PRUNABLE_TOKEN or ALL_PRUNABLE_TOKEN in self._params:
            param_names = [
                name + ".weight" for (name, _) in get_prunable_layers(module)
            ]
        else:
            param_names = self._params

        named_layers_and_params = get_named_layers_and_params_by_regex(
            module,
            param_names,
            params_strict=True,
        )

        layers = [
            named_layer_param.layer for named_layer_param in named_layers_and_params
        ]
        param_names = [
            named_layer_param.param_name
            for named_layer_param in named_layers_and_params
        ]
        layer_names = [
            named_layer_param.layer_name
            for named_layer_param in named_layers_and_params
        ]
        self._module_masks = ModuleParamPruningMask(
            layers,
            param_names,
            layer_names=layer_names,
            global_sparsity=self._global_sparsity,
        )

        self._analyzers = []
        for layer_name, layer, param_name, _ in named_layers_and_params:
            self._analyzers.append(ModulePruningAnalyzer(layer, layer_name, param_name))

        if len(self._analyzers) == 0:
            raise ValueError(
                "Could not find any params matching {} in {}".format(
                    self._params, self.__class__.__name__
                )
            )

    def update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Update the sparsity mask for the selected parameters.
        If start, enables the masks.
        If end, disables the masks if leave_enabled is False.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().update(module, optimizer, epoch, steps_per_epoch)

        if self.start_pending(epoch, steps_per_epoch):
            self._module_masks.enabled = True

        if self.end_pending(epoch, steps_per_epoch) and not self._leave_enabled:
            self._module_masks.enabled = False

        # set the mask tensors according to the new sparsity
        self._applied_sparsity = interpolate(
            epoch,
            self.start_epoch,
            self.end_epoch,
            self._init_sparsity,
            self._final_sparsity,
            self._inter_func,
        )

        self._module_masks.set_param_masks_from_sparsity(self._applied_sparsity)

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

        if (
            self._applied_sparsity != self._last_logged_sparsity
            or math.floor(epoch) != self._last_logged_epoch
        ):
            self._last_logged_sparsity = self._applied_sparsity
            self._last_logged_epoch = math.floor(epoch)
            _log_sparsity(self._analyzers, self.loggers, epoch, steps_per_epoch)

    def optimizer_post_step(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Reapply the mask after the optimizer step in case the optimizer has momentum
        that may have moved weights from 0.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().optimizer_post_step(module, optimizer, epoch, steps_per_epoch)

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

        if not isinstance(self._final_sparsity, float):
            raise TypeError(
                "final_sparsity must be of float type for {}".format(
                    self.__class__.__name__
                )
            )

        if self._final_sparsity < 0.0 or self._final_sparsity > 1.0:
            raise ValueError(
                (
                    "final_sparsity value must be in the range [0.0, 1.0],"
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
    :param final_sparsity: the final sparsity for the param to end with at end_epoch
    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param update_frequency: The number of epochs or fraction of epochs to update at
        between start and end
    :param params: A list of full parameter names or regex patterns of names to apply
        pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
        will match to all parameters. __ALL_PRUNABLE__ will match to all ConvNd
        and Linear layers' weights
    :param leave_enabled: True to continue masking the weights after end_epoch,
        False to stop masking. Should be set to False if exporting the result
        immediately after or doing some other prune
    :param inter_func: the type of interpolation function to use:
        [linear, cubic, inverse_cubic]
    :param log_types: The loggers to allow the learning rate to be logged to,
        default is __ALL__
    :param mask_type: String to define type of sparsity (options: ['unstructured',
        'channel', 'filter']), List to define block shape of a parameters in and out
        channels, or a SparsityMaskCreator object. default is 'unstructured'
    """

    def __init__(
        self,
        init_sparsity: float,
        final_sparsity: float,
        start_epoch: float,
        end_epoch: float,
        update_frequency: float,
        params: Union[str, List[str]],
        leave_enabled: bool = True,
        inter_func: str = "cubic",
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
            log_types=log_types,
            mask_type=mask_type,
            global_sparsity=False,
        )

    @ModifierProp(serializable=False)
    def global_sparsity(self) -> bool:
        """
        :return: True if global pruning is enabled, False otherwise
        """
        return self._global_sparsity


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
    |   !MagnitudePruningModifier
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
    :param log_types: The loggers to allow the learning rate to be logged to,
        default is __ALL__
    :param mask_type: String to define type of sparsity (options: ['unstructured',
        'channel', 'filter']), List to define block shape of a parameters in and out
        channels, or a SparsityMaskCreator object. default is 'unstructured'
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
            log_types=log_types,
            mask_type=mask_type,
            global_sparsity=True,
        )

    @ModifierProp(serializable=False)
    def global_sparsity(self) -> bool:
        """
        :return: True if global pruning is enabled, False otherwise
        """
        return self._global_sparsity
