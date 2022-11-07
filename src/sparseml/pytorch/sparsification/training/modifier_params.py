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
Modifier for changing the state of a modules params while training according to
certain update formulas or patterns.
"""

from typing import Any, List, Optional, Union

import torch
from torch.nn import Module, Parameter
from torch.optim.optimizer import Optimizer

from sparseml.optim import BaseModifier, ModifierProp
from sparseml.pytorch.sparsification.modifier import (
    PyTorchModifierYAML,
    ScheduledModifier,
    ScheduledUpdateModifier,
)
from sparseml.pytorch.utils import BaseLogger, get_named_layers_and_params_by_regex
from sparseml.sparsification import SparsificationTypes
from sparseml.sparsification import (
    TrainableParamsModifier as BaseTrainableParamsModifier,
)
from sparseml.utils import (
    ALL_TOKEN,
    INTERPOLATION_FUNCS,
    interpolate,
    validate_str_iterable,
)


__all__ = ["TrainableParamsModifier", "SetParamModifier", "GradualParamModifier"]


@PyTorchModifierYAML()
class TrainableParamsModifier(BaseTrainableParamsModifier, ScheduledModifier):
    """
    Modifier to control the params for a given list of parameter regex patterns.
    If end_epoch is supplied and greater than 0, then it will revert to the trainable
    settings before the modifier.
    To set all params in the given layers, set to the ALL_TOKEN string: __ALL__
    To set all layers in the given module, set to the ALL_TOKEN string: __ALL__

    | Sample yaml:
    |   !TrainableParamsModifier:
    |       params: ["conv_net.conv1.weight"]
    |       trainable: True
    |       params_strict: False
    |       start_epoch: 0
    |       end_epoch: 10

    :param params: A list of full parameter names or regex patterns of names to apply
        pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
        will match to all parameters.
    :param trainable: True if the param(s) should be made trainable,
        False to make them non-trainable
    :param params_strict: True if every regex pattern in params must match at least
        one parameter name in the module,
        False if missing params are ok and will not raise an err
    :param start_epoch: The epoch to start the modifier at
        (set to -1.0 so it starts immediately)
    :param end_epoch: The epoch to end the modifier at (set to -1.0 so it never ends),
        if > 0 then will revert to the original value for the params after this epoch
    """

    def __init__(
        self,
        params: Union[str, List[str]],
        trainable: bool,
        params_strict: bool = True,
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
    ):
        super(TrainableParamsModifier, self).__init__(
            params=params,
            trainable=trainable,
            params_strict=params_strict,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            end_comparator=-1,
        )
        self._module_params = []  # type: List[Parameter]
        self._original = []

    def initialize(
        self,
        module: Module,
        epoch: float = 0,
        loggers: Optional[List[BaseLogger]] = None,
        **kwargs,
    ):
        """
        Grab the layers params to control trainable or not for within the given module

        :param module: the PyTorch model/module to modify
        :param epoch: The epoch to initialize the modifier and module at.
            Defaults to 0 (start of the training process)
        :param loggers: Optional list of loggers to log the modification process to
        :param kwargs: Optional kwargs to support specific arguments
            for individual modifiers.
        """
        super(TrainableParamsModifier, self).initialize(
            module, epoch, loggers, **kwargs
        )
        param_names = (
            self._params
            if self._params != ALL_TOKEN and ALL_TOKEN not in self._params
            else ["re:.*"]
        )
        layers_names_and_params = get_named_layers_and_params_by_regex(
            module, param_names, params_strict=self._params_strict
        )
        for layer_name, layer, param_name, param in layers_names_and_params:
            self._module_params.append(param)

        self._check_update(epoch, steps_per_epoch=1)

    def update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        If start_pending(), updates the modules layers params to be trainable or
        not depending on given settings.
        If end_pending(), updates the modules layers params to their original
        trainable state.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().update(module, optimizer, epoch, steps_per_epoch)
        self._check_update(epoch, steps_per_epoch)

    def _enable(self, param: Parameter):
        param.requires_grad_(True)

    def _disable(self, param: Parameter):
        param.requires_grad_(False)
        param.grad = None  # clear to prevent optimizer updates

    def _check_update(self, epoch: float, steps_per_epoch: int):
        if self.start_pending(epoch, steps_per_epoch) and not self._original:
            for param in self._module_params:
                self._original.append(param.requires_grad)
                self._enable(param) if self._trainable else self._disable(param)
        elif self.end_pending(epoch, steps_per_epoch):
            for original, param in zip(self._original, self._module_params):
                self._enable(param) if original else self._disable(param)


@PyTorchModifierYAML()
class SetParamModifier(ScheduledModifier):
    """
    Modifier to set the param values for a given list of parameter name regex patterns.
    To set all parameters in the given module, set to the ALL_TOKEN string: __ALL__

    | Sample yaml:
    |   !SetParamModifier:
    |       params: ["re:.*bias"]
    |       val: [0.1, 0.1, ...]
    |       params_strict: False
    |       start_epoch: 0

    :param params: A list of full parameter names or regex patterns of names to apply
        pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
        will match to all parameters.
    :param val: The value to set for the given param in the given layers at start_epoch
    :param params_strict: True if every regex pattern in params must match at least
        one parameter name in the module,
        False if missing params are ok and will not raise an err
    :param start_epoch: The epoch to start the modifier at
        (set to -1.0 so it starts immediately)
    :param end_epoch: unused and should not be passed
    """

    def __init__(
        self,
        params: Union[str, List[str]],
        val: Any,
        params_strict: bool = True,
        start_epoch: float = 0.0,
        end_epoch: float = -1.0,
    ):
        super().__init__(
            start_epoch=start_epoch, end_epoch=end_epoch, end_comparator=None
        )
        self._params = validate_str_iterable(
            params, "{} for params".format(self.__class__.__name__)
        )
        self._val = val
        self._params_strict = params_strict
        self._module_params = []  # type: List[Parameter]

    @BaseModifier.sparsification_types.getter
    def sparsification_types(self) -> List[SparsificationTypes]:
        """
        :return: the sparsification types this modifier instance will apply
        """
        return [SparsificationTypes.general]

    @ModifierProp()
    def params(self) -> Union[str, List[str]]:
        """
        :return: A list of full parameter names or regex patterns of names to apply
            pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
            will match to all parameters.
        """
        return self._params

    @params.setter
    def params(self, value: Union[str, List[str]]):
        """
        :param value: A list of full parameter names or regex patterns of names to apply
            pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
            will match to all parameters.
        """
        self._params = validate_str_iterable(
            value, "{} for params".format(self.__class__.__name__)
        )

    @ModifierProp()
    def val(self) -> Any:
        """
        :return: The value to set for the given param in the given layers at start_epoch
        """
        return self._val

    @val.setter
    def val(self, value: Any):
        """
        :param value: The value to set for the given param in the given layers
            at start_epoch
        """
        self._val = value

    @ModifierProp()
    def params_strict(self) -> bool:
        """
        :return: True if every regex pattern in params must match at least
            one parameter name in the module,
            False if missing params are ok and will not raise an err
        """
        return self._params_strict

    @params_strict.setter
    def params_strict(self, value: bool):
        """
        :param value: True if every regex pattern in params must match at least
            one parameter name in the module,
            False if missing params are ok and will not raise an err
        """
        if self._initialized:
            raise RuntimeError(
                "Cannot change learning_rate after {} has been initialized".format(
                    self.__class__.__name__
                )
            )

        self._params_strict = value

    def initialize(
        self,
        module: Module,
        epoch: float = 0,
        loggers: Optional[List[BaseLogger]] = None,
        **kwargs,
    ):
        """
        Grab the layers params to control the values for within the given module

        :param module: the PyTorch model/module to modify
        :param epoch: The epoch to initialize the modifier and module at.
            Defaults to 0 (start of the training process)
        :param loggers: Optional list of loggers to log the modification process to
        :param kwargs: Optional kwargs to support specific arguments
            for individual modifiers.
        """
        super(SetParamModifier, self).initialize(module, epoch, loggers, **kwargs)
        param_names = (
            self._params
            if self._params != ALL_TOKEN and ALL_TOKEN not in self._params
            else ["re:.*"]
        )
        layers_names_and_params = get_named_layers_and_params_by_regex(
            module, param_names, params_strict=self._params_strict
        )

        val_tensor = torch.tensor(self._val)
        for layer_name, layer, param_name, param in layers_names_and_params:
            if param.data.shape != val_tensor.shape:
                raise ValueError(
                    "Value shape of {} does not match param shape of {}".format(
                        val_tensor.shape, param.data.shape
                    )
                )
            self._module_params.append(param)

    def update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        If start_pending(), updates the modules layers params to the
        value based on given settings.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().update(module, optimizer, epoch, steps_per_epoch)

        if self.start_pending(epoch, steps_per_epoch):
            for param in self._module_params:
                new_tens = param.data.new_tensor(self._val)
                param.data.copy_(new_tens)


@PyTorchModifierYAML()
class GradualParamModifier(ScheduledUpdateModifier):
    """
    Modifier to set the param values for a given list of parameter regex patterns
    from a start value through an end value and using an interpolation function
    for updates in between.
    To set all parameters in the given module, set to the ALL_TOKEN string: __ALL__

    | Sample YAML:
    |   !GradualParamModifier
    |       params: ["re:.*bias"]
    |       init_val: [0.0, 0.0, ...]
    |       final_val: [1.0, 1.0, ...]
    |       inter_func: linear
    |       params_strict: False
    |       start_epoch: 0.0
    |       end_epoch: 10.0
    |       update_frequency: 1.0
    """

    def __init__(
        self,
        params: Union[str, List[str]],
        init_val: Any,
        final_val: Any,
        start_epoch: float,
        end_epoch: float,
        update_frequency: float,
        inter_func: str = "linear",
        params_strict: bool = True,
    ):
        """
        :param params: A list of full parameter names or regex patterns of names
            to apply pruning to.
            Regex patterns must be specified with the prefix 're:'. __ALL__
            will match to all parameters.
        :param init_val: The initial value to set for the given param in the
            given layers at start_epoch
        :param final_val: The final value to set for the given param in the
            given layers at end_epoch
        :param start_epoch: The epoch to start the modifier at
        :param end_epoch: The epoch to end the modifier at
        :param update_frequency: The number of epochs or fraction of epochs to
            update at between start and end
        :param inter_func: the type of interpolation function to use:
            [linear, cubic, inverse_cubic]; default is linear
        :param params_strict: True if every regex pattern in params must match at least
            one parameter name in the module
            False if missing params are ok -- will not raise an err
        """
        super().__init__(
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            min_end=0.0,
            end_comparator=1,
        )
        self._params = validate_str_iterable(
            params, "{} for params".format(self.__class__.__name__)
        )
        self._init_val = init_val
        self._final_val = final_val
        self._init_val_tens = None
        self._final_val_tens = None
        self._inter_func = inter_func
        self._params_strict = params_strict
        self._module_params = []  # type: List[Parameter]

        self.validate()

    @BaseModifier.sparsification_types.getter
    def sparsification_types(self) -> List[SparsificationTypes]:
        """
        :return: the sparsification types this modifier instance will apply
        """
        return [SparsificationTypes.general]

    @ModifierProp()
    def params(self) -> Union[str, List[str]]:
        """
        :return: A list of full parameter names or regex patterns of names to apply
            pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
            will match to all parameters.
        """
        return self._params

    @params.setter
    def params(self, value: Union[str, List[str]]):
        """
        :param value: A list of full parameter names or regex patterns of names to apply
            pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
            will match to all parameters.
        """
        self._params = validate_str_iterable(
            value, "{} for params".format(self.__class__.__name__)
        )

    @ModifierProp()
    def init_val(self) -> Any:
        """
        :return: The initial value to set for the given param in the given layers
            at start_epoch
        """
        return self._init_val

    @init_val.setter
    def init_val(self, value: Any):
        """
        :param value: The initial value to set for the given param in the given layers
            at start_epoch
        """
        self._init_val = value

    @ModifierProp()
    def final_val(self) -> Any:
        """
        :return: The final value to set for the given param in the given layers at
            end_epoch
        """
        return self._final_val

    @final_val.setter
    def final_val(self, value: Any):
        """
        :param value: The final value to set for the given param in the given layers
            at end_epoch
        """
        self._final_val = value

    @ModifierProp()
    def inter_func(self) -> str:
        """
        :return: the type of interpolation function to use:
            [linear, cubic, inverse_cubic]; default is linear
        """
        return self._inter_func

    @inter_func.setter
    def inter_func(self, value: str):
        """
        :param value: the type of interpolation function to use:
            [linear, cubic, inverse_cubic]; default is linear
        """
        self._inter_func = value
        self.validate()

    @ModifierProp()
    def params_strict(self) -> bool:
        """
        :return: True if every regex pattern in params must match at least
            one parameter name in the module
            False if missing params are ok -- will not raise an err
        """
        return self._params_strict

    @params_strict.setter
    def params_strict(self, value: bool):
        """
        :param value: True if every regex pattern in params must match at least
            one parameter name in the module
            False if missing params are ok -- will not raise an err
        """
        self._params_strict = value

    def initialize(
        self,
        module: Module,
        epoch: float = 0,
        loggers: Optional[List[BaseLogger]] = None,
        **kwargs,
    ):
        """
        Grab the layers params to control the values for within the given module

        :param module: the PyTorch model/module to modify
        :param epoch: The epoch to initialize the modifier and module at.
            Defaults to 0 (start of the training process)
        :param loggers: Optional list of loggers to log the modification process to
        :param kwargs: Optional kwargs to support specific arguments
            for individual modifiers.
        """
        super(GradualParamModifier, self).initialize(module, epoch, loggers, **kwargs)

        self._init_val_tens = torch.tensor(self._init_val)
        self._final_val_tens = torch.tensor(self._final_val)

        if self._init_val_tens.shape != self._final_val_tens.shape:
            raise ValueError(
                "init_val shape {} must match final_val shape {}".format(
                    self._init_val_tens.shape, self._final_val_tens.shape
                )
            )

        param_names = (
            self._params
            if self._params != ALL_TOKEN and ALL_TOKEN not in self._params
            else ["re:.*"]
        )
        layers_names_and_params = get_named_layers_and_params_by_regex(
            module, param_names, params_strict=self._params_strict
        )

        for layer_name, layer, param_name, param in layers_names_and_params:
            if param.data.shape != self._init_val_tens.shape:
                raise ValueError(
                    "Value shape of {} does not match param shape of {}".format(
                        self._init_val_tens.shape, param.data.shape
                    )
                )
            self._module_params.append(param)

    def update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Updates the modules layers params to the interpolated value based on given
        settings and current epoch.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().update(module, optimizer, epoch, steps_per_epoch)

        new_val = interpolate(
            epoch,
            self.start_epoch,
            self.end_epoch,
            self._init_val_tens,
            self._final_val_tens,
        )

        for param in self._module_params:
            new_tens = param.data.new_tensor(new_val)
            param.data.copy_(new_tens)

    def validate(self):
        """
        Validate the values of the params for the current instance are valid
        """

        if self._inter_func not in INTERPOLATION_FUNCS:
            raise ValueError(
                (
                    "{} is not a supported inter_func in layers_settings,"
                    " available are {} for {}"
                ).format(self._inter_func, INTERPOLATION_FUNCS, self.__class__.__name__)
            )
