"""
Modifier for changing the state of a module's params while training according to
certain update formulas or patterns.
"""

from typing import List, Union, Any
import torch
from torch.nn import Module, Parameter
from torch.optim.optimizer import Optimizer

from neuralmagicML.utils import (
    ALL_TOKEN,
    convert_to_bool,
    validate_str_iterable,
    interpolate,
    INTERPOLATION_FUNCS,
)
from neuralmagicML.recal import ModifierProp
from neuralmagicML.pytorch.recal.modifier import (
    PyTorchModifierYAML,
    ScheduledModifier,
    ScheduledUpdateModifier,
)
from neuralmagicML.pytorch.utils import (
    get_layer,
    get_terminal_layers,
)


__all__ = ["TrainableParamsModifier", "SetParamModifier", "GradualParamModifier"]


@PyTorchModifierYAML()
class TrainableParamsModifier(ScheduledModifier):
    """
    Modifier to control the params for a given list of layers to apply the trainable or
    not (requires_grad var).
    If end_epoch is supplied and greater than 0, then it will revert to the trainable
    settings before the modifier.
    To set all params in the given layers, set to the ALL_TOKEN string: __ALL__
    To set all layers in the given module, set to the ALL_TOKEN string: __ALL__

    | Sample yaml:
    |   !TrainableParamsModifier:
    |       params:
    |           - weight
    |           - bias
    |       layers: __ALL__
    |       trainable: True
    |       params_strict: False
    |       start_epoch: 0
    |       end_epoch: 10

    :param params: str or list of str for the params to apply the trainable modifier to
        can also use the token __ALL__ to specify all params
    :param layers: str or list of str for the layers to apply the trainable modifier to
        can also use the token __ALL__ to specify all layers
    :param trainable: True if the param(s) should be made trainable,
        False to make them non-trainable
    :param params_strict: True if the given param(s) must be found in each layer
        -- will raise an err if not found,
        False if missing params are ok -- will not raise an err
    :param start_epoch: The epoch to start the modifier at
        (set to -1.0 so it starts immediately)
    :param end_epoch: The epoch to end the modifier at (set to -1.0 so it never ends),
        if > 0 then will revert to the original value for the params after this epoch
    """

    def __init__(
        self,
        params: Union[str, List[str]],
        layers: Union[str, List[str]],
        trainable: bool,
        params_strict: bool = True,
        start_epoch: float = 0.0,
        end_epoch: float = -1.0,
    ):
        super().__init__(
            start_epoch=start_epoch, end_epoch=end_epoch, end_comparator=-1
        )
        self._start_epoch = start_epoch
        self._params = validate_str_iterable(
            params, "{} for params".format(self.__class__.__name__)
        )
        self._layers = validate_str_iterable(
            layers, "{} for layers".format(self.__class__.__name__)
        )
        self._trainable = convert_to_bool(trainable)
        self._params_strict = convert_to_bool(params_strict)
        self._module_params = []  # type: List[Parameter]
        self._original = []

    @ModifierProp()
    def params(self) -> Union[str, List[str]]:
        """
        :return: str or list of str for the params to apply the trainable modifier to.
            Can also use the token __ALL__ to specify all params
        """
        return self._params

    @params.setter
    def params(self, value: Union[str, List[str]]):
        """
        :param value: str or list of str for the params to apply the trainable modifier
            to.Can also use the token __ALL__ to specify all params
        """
        self._params = validate_str_iterable(
            value, "{} for params".format(self.__class__.__name__)
        )

    @ModifierProp()
    def layers(self) -> Union[str, List[str]]:
        """
        :return: str or list of str for the layers to apply the trainable modifier to.
            Can also use the token __ALL__ to specify all layers
        """
        return self._layers

    @layers.setter
    def layers(self, value: Union[str, List[str]]):
        """
        :param value: str or list of str for the layers to apply the trainable modifier
            to. Can also use the token __ALL__ to specify all layers
        """
        self._layers = validate_str_iterable(
            value, "{} for layers".format(self.__class__.__name__)
        )

    @ModifierProp()
    def trainable(self) -> bool:
        """
        :return: True if the param(s) should be made trainable,
            False to make them non-trainable
        """
        return self._trainable

    @trainable.setter
    def trainable(self, value: bool):
        """
        :param value: True if the param(s) should be made trainable,
            False to make them non-trainable
        """
        self._trainable = value

    @ModifierProp()
    def params_strict(self) -> bool:
        """
        :return: True if the given param(s) must be found in each layer
            -- will raise an err if not found.
            False if missing params are ok -- will not raise an err
        """
        return self._params_strict

    @params_strict.setter
    def params_strict(self, value: bool):
        """
        :param value: True if the given param(s) must be found in each layer
            -- will raise an err if not found.
            False if missing params are ok -- will not raise an err
        """
        self._params_strict = value

    def initialize(self, module: Module, optimizer: Optimizer):
        """
        Grab the layers' params to control trainable or not for within the given module

        :param module: module to modify
        :param optimizer: optimizer to modify
        """
        super(TrainableParamsModifier, self).initialize(module, optimizer)
        layers = (
            get_terminal_layers(module)
            if self._layers == ALL_TOKEN
            else {name: get_layer(name, module) for name in self._layers}
        )

        for name, layer in layers.items():
            found = []

            for param_name, param in layer.named_parameters():
                if self._params == ALL_TOKEN or param_name in self._params:
                    found.append(param_name)
                    self._module_params.append(param)

            if (
                self._params_strict
                and self._params != ALL_TOKEN
                and len(found) != len(self._params)
            ):
                raise ValueError(
                    (
                        "Could not find all required params for layer {}"
                        " with params {} and required {} for {}"
                    ).format(layer, found, self._params, self.__class__.__name__)
                )

    def update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        If start_pending(), updates the module's layers' params to be trainable or
        not depending on given settings.
        If end_pending(), updates the module's layers' params to their original
        trainable state.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().update(module, optimizer, epoch, steps_per_epoch)

        if self.start_pending(epoch, steps_per_epoch):
            self._original.clear()

            for param in self._module_params:
                self._original.append(param.requires_grad)
                param.requires_grad = self._trainable
        elif self.end_pending(epoch, steps_per_epoch):
            for original, param in zip(self._original, self._module_params):
                param.requires_grad = original


@PyTorchModifierYAML()
class SetParamModifier(ScheduledModifier):
    """
    Modifier to set the param values for a given list of layers.
    To set all layers in the given module, set to the ALL_TOKEN string: __ALL__

    | Sample yaml:
    |   !SetParamModifier:
    |       param: bias
    |       layers: __ALL__
    |       val: [0.1, 0.1, ...]
    |       params_strict: False
    |       start_epoch: 0

    :param param: name of the param to apply the given value for
    :param layers: str or list of str for the layers to apply the given value for
        can also use the token __ALL__ to specify all layers
    :param val: The value to set for the given param in the given layers at start_epoch
    :param param_strict: True if the given param must be found in each layer
        -- will raise an err if not found.
        False if missing params are ok -- will not raise an err
    :param start_epoch: The epoch to start the modifier at
        (set to -1.0 so it starts immediately)
    :param end_epoch: unused and should not be passed
    """

    def __init__(
        self,
        param: str,
        layers: Union[str, List[str]],
        val: Any,
        param_strict: bool = True,
        start_epoch: float = 0.0,
        end_epoch: float = -1.0,
    ):
        super().__init__(
            start_epoch=start_epoch, end_epoch=end_epoch, end_comparator=None
        )
        self._param = param
        self._val = val
        self._layers = validate_str_iterable(
            layers, "{} for layers".format(self.__class__.__name__)
        )
        self._param_strict = param_strict
        self._module_params = []  # type: List[Parameter]

    @ModifierProp()
    def param(self) -> str:
        """
        :return: name of the param to apply the given value for
        """
        return self._param

    @param.setter
    def param(self, value: str):
        """
        :param value: name of the param to apply the given value for
        """
        self._param = value

    @ModifierProp()
    def layers(self) -> Union[str, List[str]]:
        """
        :return: str or list of str for the layers to apply the given value for
                 can also use the token __ALL__ to specify all layers
        """
        return self._layers

    @layers.setter
    def layers(self, value: Union[str, List[str]]):
        """
        :param value: str or list of str for the layers to apply the given value for
                      can also use the token __ALL__ to specify all layers
        """
        self._layers = validate_str_iterable(
            value, "{} for layers".format(self.__class__.__name__)
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
    def param_strict(self) -> bool:
        """
        :return: True if the given param must be found in each layer
            -- will raise an err if not found.
            False if missing params are ok -- will not raise an err
        """
        return self._param_strict

    @param_strict.setter
    def param_strict(self, value: bool):
        """
        :param value: True if the given param must be found in each layer
            -- will raise an err if not found.
            False if missing params are ok -- will not raise an err
        """
        if self._initialized:
            raise RuntimeError(
                "Cannot change learning_rate after {} has been initialized".format(
                    self.__class__.__name__
                )
            )

        self._param_strict = value

    def initialize(self, module: Module, optimizer: Optimizer):
        """
        Grab the layers' params to control the values for within the given module

        :param module: module to modify
        :param optimizer: optimizer to modify
        """
        super(SetParamModifier, self).initialize(module, optimizer)
        layers = (
            get_terminal_layers(module)
            if self._layers == ALL_TOKEN
            else {name: get_layer(name, module) for name in self._layers}
        )

        for name, layer in layers.items():
            found = False

            for param_name, par in layer.named_parameters():
                if param_name == self._param:
                    val_tensor = torch.tensor(self._val)

                    if par.data.shape != val_tensor.shape:
                        raise ValueError(
                            "Value shape of {} does not match param shape of {}".format(
                                val_tensor.shape, par.data.shape
                            )
                        )

                    self._module_params.append(par)
                    found = True
                    break

            if self._param_strict and self._layers != ALL_TOKEN and not found:
                raise ValueError(
                    "Could not find required param {} in layer {} for {}".format(
                        self._param, layer, self.__class__.__name__
                    )
                )

    def update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        If start_pending(), updates the module's layers' params to the
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
    Modifier to set the param values for a given list of layers from a start value
    through an end value and using an interpolation function for updates in between.
    To set all layers in the given module, set to the ALL_TOKEN string: __ALL__

    | Sample YAML:
    |   !GradualParamModifier
    |       param: bias
    |       layers: __ALL__
    |       init_val: [0.0, 0.0, ...]
    |       final_val: [1.0, 1.0, ...]
    |       inter_func: linear
    |       param_strict: False
    |       start_epoch: 0.0
    |       end_epoch: 10.0
    |       update_frequency: 1.0
    """

    def __init__(
        self,
        param: str,
        layers: Union[str, List[str]],
        init_val: Any,
        final_val: Any,
        start_epoch: float,
        end_epoch: float,
        update_frequency: float,
        inter_func: str = "linear",
        param_strict: bool = True,
    ):
        """
        :param param: name of the param to apply the given value for
        :param layers: str or list of str for the layers to apply the given value for
            can also use the token __ALL__ to specify all layers
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
        :param param_strict: True if the given param must be found in each layer
            -- will raise an err if not found.
            False if missing params are ok -- will not raise an err; default is True
        """
        super().__init__(
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            min_end=0.0,
            end_comparator=1,
        )
        self._param = param
        self._init_val = init_val
        self._final_val = final_val
        self._init_val_tens = None
        self._final_val_tens = None
        self._layers = validate_str_iterable(
            layers, "{} for layers".format(self.__class__.__name__)
        )
        self._inter_func = inter_func
        self._param_strict = param_strict
        self._module_params = []  # type: List[Parameter]

        self.validate()

    @ModifierProp()
    def param(self) -> str:
        """
        :return: name of the param to apply the given value for
        """
        return self._param

    @param.setter
    def param(self, value: str):
        """
        :param value: name of the param to apply the given value for
        """
        self._param = value

    @ModifierProp()
    def layers(self) -> Union[str, List[str]]:
        """
        :return: str or list of str for the layers to apply the given value for
            can also use the token __ALL__ to specify all layers
        """
        return self._layers

    @layers.setter
    def layers(self, value: Union[str, List[str]]):
        """
        :param value: str or list of str for the layers to apply the given value for
            can also use the token __ALL__ to specify all layers
        """
        self._layers = validate_str_iterable(
            value, "{} for layers".format(self.__class__.__name__)
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
    def param_strict(self) -> bool:
        """
        :return: True if the given param must be found in each layer
            -- will raise an err if not found.
            False if missing params are ok -- will not raise an err; default is True
        """
        return self._param_strict

    @param_strict.setter
    def param_strict(self, value: bool):
        """
        :param value: True if the given param must be found in each layer
            -- will raise an err if not found.
            False if missing params are ok -- will not raise an err; default is True
        """
        self._param_strict = value

    def initialize(self, module: Module, optimizer: Optimizer):
        """
        Grab the layers' params to control the values for within the given module

        :param module: module to modify
        :param optimizer: optimizer to modify
        """
        super(GradualParamModifier, self).initialize(module, optimizer)

        self._init_val_tens = torch.tensor(self._init_val)
        self._final_val_tens = torch.tensor(self._final_val)

        if self._init_val_tens.shape != self._final_val_tens.shape:
            raise ValueError(
                "init_val shape {} must match final_val shape {}".format(
                    self._init_val_tens.shape, self._final_val_tens.shape
                )
            )

        layers = (
            get_terminal_layers(module)
            if self._layers == ALL_TOKEN
            else {name: get_layer(name, module) for name in self._layers}
        )

        for name, layer in layers.items():
            found = False

            for param_name, par in layer.named_parameters():
                if param_name == self._param:
                    if par.data.shape != self._init_val_tens.shape:
                        raise ValueError(
                            "Value shape of {} does not match param shape of {}".format(
                                self._init_val_tens.shape, par.data.shape
                            )
                        )

                    self._module_params.append(par)
                    found = True
                    break

            if self._param_strict and self._layers != ALL_TOKEN and not found:
                raise ValueError(
                    "Could not find required param {} in layer {} for {}".format(
                        self._param, layer, self.__class__.__name__
                    )
                )

    def update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Updates the module's layers' params to the interpolated value based on given
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
