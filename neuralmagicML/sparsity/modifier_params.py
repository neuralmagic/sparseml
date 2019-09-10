from typing import List, Union, Any
import yaml
from torch.nn import Module, Parameter
from torch.optim.optimizer import Optimizer

from .modifier import ScheduledModifier, ScheduledUpdateModifier, ALL_TOKEN
from .utils import convert_to_bool, validate_str_list, get_layer, get_terminal_layers, interpolate, INTERPOLATION_FUNCS


__all__ = ['TrainableParamsModifier', 'SetParamModifier', 'GradualParamModifier']


class TrainableParamsModifier(ScheduledModifier):
    YAML_KEY = u'!TrainableParamsModifier'

    @staticmethod
    def yaml_constructor(loader, node):
        instance = TrainableParamsModifier.__new__(TrainableParamsModifier)
        yield instance
        state = loader.construct_mapping(node, deep=True)
        instance.__init__(**state)

    def __init__(self, params: Union[str, List[str]], layers: Union[str, List[str]],
                 trainable: bool, params_strict: bool = True, start_epoch: float = 0.0, end_epoch: float = -1.0):
        """
        Controls whether given params in given layers are considered trainable or not (requires_grad var)

        Sample yaml:
            !TrainableParamsModifier:
                params:
                    - weight
                    - bias
                layers: __ALL__
                trainable: True
                params_strict: False
                start_epoch: 0
                end_epoch: 10

        :param params: str or list of str for the params to apply the trainable modifier to
                       can also use the token __ALL__ to specify all params
        :param layers: str or list of str for the layers to apply the trainable modifier to
                       can also use the token __ALL__ to specify all layers
        :param trainable: True if the param(s) should be made trainable, false to make them non-trainable
        :param params_strict: True if the given param(s) must be found in each layer -- will raise an err if not found,
                              False if missing params are ok -- will not raise an err
        :param start_epoch: The epoch to start the modifier at (set to -1.0 so it starts immediately)
        :param end_epoch: The epoch to end the modifier at (set to -1.0 so it never ends)
                          here for mainly tracking, will not apply anything at end, modifiers easily start to conflict
        """
        super().__init__(start_epoch, end_epoch)
        self._start_epoch = start_epoch
        self._params = validate_str_list(params, 'params', self.__class__.__name__)
        self._layers = validate_str_list(layers, 'layers', self.__class__.__name__)
        self._trainable = convert_to_bool(trainable)
        self._params_strict = convert_to_bool(params_strict)
        self._module_params = []  # type: List[Parameter]

    def initialize(self, module: Module, optimizer: Optimizer):
        """
        Grab the layers' params to control trainable or not for within the given module

        :param module: module to modify
        :param optimizer: optimizer to modify
        """
        super(TrainableParamsModifier, self).initialize(module, optimizer)
        layers = get_terminal_layers(module) if self._layers == ALL_TOKEN else \
            [get_layer(name, module) for name in self._layers]

        for layer in layers:
            found = []

            for param_name, param in layer.named_parameters():
                if self._params == ALL_TOKEN or param_name in self._params:
                    found.append(param_name)
                    self._module_params.append(param)

            if self._params_strict and self._params != ALL_TOKEN and len(found) != len(self._params):
                raise ValueError('Could not find all required params for layer {} with params {} and required {} for {}'
                                 .format(layer, found, self._params, self.__class__.__name__))

    def update(self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int):
        """
        If start_pending(), updates the module's layers' params to be trainable or not depending on given settings

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """
        if self.start_pending(epoch, steps_per_epoch):
            for param in self._module_params:
                param.requires_grad = self._trainable


yaml.add_constructor(TrainableParamsModifier.YAML_KEY, TrainableParamsModifier.yaml_constructor)
yaml.add_constructor(TrainableParamsModifier.YAML_KEY, TrainableParamsModifier.yaml_constructor, yaml.SafeLoader)


class SetParamModifier(ScheduledModifier):
    """
    Sample YAML:

    !SetParamPolicy:
      start_epoch: 0.0
      param: threshold
      val: 0.0
      layers:
        - layer1.0.relu1
        - layer1.1.relu2
    """

    YAML_KEY = u'!SetParamModifier'

    @staticmethod
    def yaml_constructor(loader, node):
        instance = SetParamModifier.__new__(SetParamModifier)
        yield instance
        state = loader.construct_mapping(node, deep=True)
        instance.__init__(**state)

    def __init__(self, param: str, layers: Union[str, List[str]], val: Any,
                 param_strict: bool = True, start_epoch: float = 0.0, end_epoch: float = -1.0):
        """
        Controls whether given params in given layers are considered trainable or not (requires_grad var)

        Sample yaml:
            !SetParamModifier:
                params:
                    - bias
                layers: __ALL__
                val: [0.1, 0.1, ...]
                params_strict: False
                start_epoch: 0
                end_epoch: 10

        :param param: name of the param to apply the given value for
        :param layers: str or list of str for the layers to apply the given value for
                       can also use the token __ALL__ to specify all layers
        :param val: The value to set for the given param in the given layers at start_epoch
        :param param_strict: True if the given param must be found in each layer -- will raise an err if not found,
                             False if missing params are ok -- will not raise an err
        :param start_epoch: The epoch to start the modifier at (set to -1.0 so it starts immediately)
        :param end_epoch: The epoch to end the modifier at (set to -1.0 so it never ends)
                          here for mainly tracking, will not apply anything at end, modifiers easily start to conflict
        """
        super().__init__(start_epoch, end_epoch)
        self._param = param
        self._val = val
        self._layers = validate_str_list(layers, 'layers', self.__class__.__name__)
        self._param_strict = param_strict
        self._module_params = []  # type: List[Parameter]

    def initialize(self, module: Module, optimizer: Optimizer):
        """
        Grab the layers' params to control the values for within the given module

        :param module: module to modify
        :param optimizer: optimizer to modify
        """
        super(SetParamModifier, self).initialize(module, optimizer)
        layers = get_terminal_layers(module) if self._layers == ALL_TOKEN else \
            [get_layer(name, module) for name in self._layers]

        for layer in layers:
            found = False

            for param_name, par in layer.named_parameters():
                if param_name == self._param:
                    if not par.shape and not isinstance(self._val, float):
                        raise TypeError('Cannot apply a type {} to layer {} float param {} for {}'
                                        .format(type(self._val), layer, param_name, self.__class__.__name__))
                    elif par.shape and not isinstance(self._val, List):
                            raise TypeError('Cannot apply a type {} to layer {} list param {} for {}'
                                            .format(type(self._val), layer, param_name, self.__class__.__name__))
                    elif par.shape and len(par) != len(self._val):
                            raise ValueError('Cannot apply a val len {} to layer {} param {} with len {} for {}'
                                             .format(len(self._val), layer, param_name, len(par),
                                                     self.__class__.__name__))

                    self._module_params.append(par)
                    found = True
                    break

            if self._param_strict and self._layers != ALL_TOKEN and not found:
                raise ValueError('Could not find required param {} in layer {} for {}'
                                .format(self._param, layer, self.__class__.__name__))

    def update(self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int):
        """
        If start_pending(), updates the module's layers' params to the value based on given settings

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """
        if self.start_pending(epoch, steps_per_epoch):
            for param in self._module_params:
                new_tens = param.data.new_tensor(self._val)
                param.data.copy_(new_tens)


yaml.add_constructor(SetParamModifier.YAML_KEY, SetParamModifier.yaml_constructor)
yaml.add_constructor(SetParamModifier.YAML_KEY, SetParamModifier.yaml_constructor, yaml.SafeLoader)


class GradualParamModifier(ScheduledUpdateModifier):
    """
    Sample YAML:

    !GradualParamPolicy:
      start_epoch: 0.0
      end_epoch: 10.0
      update_frequency: 2.0
      param: threshold
      init_val: 0.0
      final_val: 1.0
      inter_func: cubic
      layers:
        - layer1.0.relu1
        - layer1.1.relu2
    """

    YAML_KEY = u'!GradualParamModifier'

    @staticmethod
    def yaml_constructor(loader, node):
        instance = GradualParamModifier.__new__(GradualParamModifier)
        yield instance
        state = loader.construct_mapping(node, deep=True)
        instance.__init__(**state)

    def __init__(self, param: str, layers: Union[str, List[str]],
                 init_val: Union[float, List[float]], final_val: Union[float, List[float]],
                 inter_func: str = 'linear', param_strict: bool = True,
                 start_epoch: float = -1.0, end_epoch: float = -1.0, update_frequency: float = -1.0):
        """
        Controls a schedule for setting the value of a param over a certain number of epochs
        Uses an interpolation function to gradually move from init_val to final_val

        Sample yaml:
            !GradualParamModifier
                params:
                    - bias
                layers: __ALL__
                init_val: [0.0, 0.0, ...]
                final_val: [1.0, 1.0, ...]
                inter_func: linear
                param_strict: False
                start_epoch: 0.0
                end_epoch: 10.0
                update_frequency: 1.0

        :param param: name of the param to apply the given value for
        :param layers: str or list of str for the layers to apply the given value for
                       can also use the token __ALL__ to specify all layers
        :param init_val: The initial value to set for the given param in the given layers at start_epoch
        :param final_val: The final value to set for the given param in the given layers at end_epoch
        :param inter_func: the type of interpolation function to use: [linear, cubic, inverse_cubic]
        :param param_strict: True if the given param must be found in each layer -- will raise an err if not found,
                             False if missing params are ok -- will not raise an err
        :param start_epoch: The epoch to start the modifier at
        :param end_epoch: The epoch to end the modifier at
        :param update_frequency: The number of epochs or fraction of epochs to update at between start and end
        """
        super().__init__(start_epoch, end_epoch, update_frequency)
        self._param = param
        self._init_val = init_val
        self._final_val = final_val
        self._layers = validate_str_list(layers, 'layers', self.__class__.__name__)
        self._inter_func = inter_func
        self._param_strict = param_strict
        self._module_params = []  # type: List[Parameter]

        if start_epoch < 0:
            raise ValueError('start_epoch must be greater than or equal to 0 for {}'.format(self.__class__.__name__))

        if end_epoch < 0:
            raise ValueError('end_epoch must be greater than or equal to 0 for {}'.format(self.__class__.__name__))

        if update_frequency <= 0:
            raise ValueError('update_frequency must be greater than 0 for {}'.format(self.__class__.__name__))

        if type(self._init_val) != type(self._final_val):
            raise TypeError('init_val of type {} does not match final_val type of {} for {}'
                            .format(type(self._init_val), type(self._final_val), self.__class__.__name__))

        if isinstance(self._init_val, List) and len(self._init_val) != len(self._final_val):
            raise ValueError('init_val of len {} does not match final_val len of {} for {}'
                             .format(len(self._init_val), len(self._final_val), self.__class__.__name__))

        if self._inter_func not in INTERPOLATION_FUNCS:
            raise ValueError('{} is not a supported inter_func in layers_settings, available are {} for {}'
                             .format(self._inter_func, INTERPOLATION_FUNCS, self.__class__.__name__))

    def initialize(self, module: Module, optimizer: Optimizer):
        """
        Grab the layers' params to control the values for within the given module

        :param module: module to modify
        :param optimizer: optimizer to modify
        """
        super(GradualParamModifier, self).initialize(module, optimizer)
        layers = get_terminal_layers(module) if self._layers == ALL_TOKEN else \
            [get_layer(name, module) for name in self._layers]

        for layer in layers:
            found = False

            for param_name, par in layer.named_parameters():
                if param_name == self._param:
                    if not par.shape and not isinstance(self._init_val, float):
                        raise TypeError('Cannot apply a type {} to layer {} float param {} for {}'
                                        .format(type(self._init_val), layer, param_name, self.__class__.__name__))
                    elif par.shape and not isinstance(self._init_val, List):
                        raise TypeError('Cannot apply a type {} to layer {} list param {} for {}'
                                        .format(type(self._init_val), layer, param_name, self.__class__.__name__))
                    elif par.shape and len(par) != len(self._init_val):
                        raise ValueError('Cannot apply a val len {} to layer {} param {} with len {} for {}'
                                         .format(len(self._init_val), layer, param_name, len(par),
                                                 self.__class__.__name__))

                    self._module_params.append(par)
                    found = True
                    break

            if self._param_strict and self._layers != ALL_TOKEN and not found:
                raise ValueError('Could not find required param {} in layer {} for {}'
                                 .format(self._param, layer, self.__class__.__name__))

    def update(self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int):
        """
        Updates the module's layers' params to the interpolated value based on given settings and current epoch

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """
        if not isinstance(self._init_val, List):
            new_val = interpolate(epoch, self.start_epoch, self.end_epoch, self._init_val,
                                  self._final_val, self._inter_func)
        else:
            new_val = [interpolate(epoch, self.start_epoch, self.end_epoch, init, final, self._inter_func)
                       for init, final in zip(self._init_val, self._final_val)]

        for param in self._module_params:
            new_tens = param.data.new_tensor(new_val)
            param.data.copy_(new_tens)


yaml.add_constructor(GradualParamModifier.YAML_KEY, GradualParamModifier.yaml_constructor)
yaml.add_constructor(GradualParamModifier.YAML_KEY, GradualParamModifier.yaml_constructor, yaml.SafeLoader)
