from typing import Union, List
import yaml
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from ...utils import (
    INTERPOLATION_FUNCS, get_terminal_layers, get_layer, convert_to_bool, interpolate, validate_str_list
)
from ..modifier import ScheduledUpdateModifier
from ..helpers import ALL_TOKEN
from .mask import KSLayerParamMask


__all__ = ['GradualKSModifier']


class GradualKSModifier(ScheduledUpdateModifier):
    YAML_KEY = u'!GradualKSModifier'

    @staticmethod
    def yaml_constructor(loader, node):
        instance = GradualKSModifier.__new__(GradualKSModifier)
        yield instance
        state = loader.construct_mapping(node, deep=True)
        instance.__init__(**state)

    def __init__(self, param: str, layers: Union[str, List[str]], init_sparsity: float, final_sparsity: float,
                 leave_enabled: bool = True, inter_func: str = 'linear', param_strict: bool = True,
                 start_epoch: float = -1.0, end_epoch: float = -1.0, update_frequency: float = -1.0):
        """
        Gradually applies kernel sparsity to a given layer or layers from init_sparsity until final_sparsity is reached
        over a given amount of time and applied with an interpolated function for each step taken

        Applies based on magnitude pruning without any structure to the pruning

        Sample yaml:
            !GradualKSModifier
                param: weight
                layers: __ALL__
                init_sparsity: 0.05
                final_sparsity: 0.8
                prune_global: False
                leave_enabled: True
                inter_func: cubic
                param_strict: False
                start_epoch: 0.0
                end_epoch: 10.0
                update_frequency: 1.0

        :param param: the name of the parameter to apply pruning to, generally 'weight' for linear and convs
        :param layers: str or list of str for the layers to apply the KS modifier to
                       can also use the token __ALL__ to specify all layers
        :param init_sparsity: the initial sparsity for the param to start with at start_epoch
        :param final_sparsity: the final sparsity for the param to end with at end_epoch
        :param leave_enabled: True to continue masking the weights after end_epoch, False to stop masking
                              Should be set to False if exporting the result immediately after or doing some other prune
        :param inter_func: the type of interpolation function to use: [linear, cubic, inverse_cubic]
        :param param_strict: True if the given param must be found in each layer -- will raise an err if not found,
                             False if missing params are ok -- will not raise an err
        :param start_epoch: The epoch to start the modifier at
        :param end_epoch: The epoch to end the modifier at
        :param update_frequency: The number of epochs or fraction of epochs to update at between start and end
        """
        super().__init__(start_epoch, end_epoch, update_frequency)
        self._param = param
        self._layers = validate_str_list(layers, 'layers', GradualKSModifier.YAML_KEY)
        self._init_sparsity = init_sparsity
        self._final_sparsity = final_sparsity
        self._leave_enabled = convert_to_bool(leave_enabled)
        self._inter_func = inter_func
        self._param_strict = convert_to_bool(param_strict)
        self._module_masks = []  # type: List[KSLayerParamMask]

        if not isinstance(self._init_sparsity, float):
            raise TypeError('init_sparsity must be of float type for {}'.format(self.__class__.__name__))

        if self._init_sparsity < 0.0 or self._init_sparsity > 1.0:
            raise ValueError('init_sparsity value must be in the range [0.0, 1.0], given {} for {}'
                             .format(self._init_sparsity, self.__class__.__name__))

        if not isinstance(self._final_sparsity, float):
            raise TypeError('final_sparsity must be of float type for {}'.format(self.__class__.__name__))

        if self._final_sparsity < 0.0 or self._final_sparsity > 1.0:
            raise ValueError('init_sparsity value must be in the range [0.0, 1.0], given {} for {}'
                             .format(self._init_sparsity, self.__class__.__name__))

        if self._inter_func not in INTERPOLATION_FUNCS:
            raise ValueError('{} is not a supported inter_func in layers_settings, available are {} for {}'
                             .format(self._inter_func, INTERPOLATION_FUNCS, self.__class__.__name__))

    def __del__(self):
        self._module_masks.clear()
        
    @property
    def param(self) -> str:
        return self._param
    
    @param.setter
    def param(self, value: str):
        if self.initialized:
            raise RuntimeError('Cannot change param after {} has been initialized'.format(self.__class__.__name__))
        
        self._param = value

    @property
    def layers(self) -> Union[str, List[str]]:
        return self._layers

    @layers.setter
    def layers(self, value: Union[str, List[str]]):
        if self.initialized:
            raise RuntimeError('Cannot change layers after {} has been initialized'
                               .format(self.__class__.__name__))

        self._layers = value

    @property
    def init_sparsity(self) -> float:
        return self._init_sparsity

    @init_sparsity.setter
    def init_sparsity(self, value: float):
        if self.initialized:
            raise RuntimeError('Cannot change init_sparsity after {} has been initialized'
                               .format(self.__class__.__name__))

        self._init_sparsity = value

    @property
    def final_sparsity(self) -> float:
        return self._final_sparsity

    @final_sparsity.setter
    def final_sparsity(self, value: float):
        if self.initialized:
            raise RuntimeError('Cannot change final_sparsity after {} has been initialized'
                               .format(self.__class__.__name__))

        self._final_sparsity = value

    @property
    def leave_enabled(self) -> bool:
        return self._leave_enabled

    @leave_enabled.setter
    def leave_enabled(self, value: bool):
        if self.initialized:
            raise RuntimeError('Cannot change leave_enabled after {} has been initialized'
                               .format(self.__class__.__name__))

        self._leave_enabled = value

    @property
    def inter_func(self) -> str:
        return self._inter_func

    @inter_func.setter
    def inter_func(self, value: str):
        if self.initialized:
            raise RuntimeError('Cannot change inter_func after {} has been initialized'
                               .format(self.__class__.__name__))

        self._inter_func = value

    @property
    def param_strict(self) -> float:
        return self._param_strict

    @param_strict.setter
    def param_strict(self, value: float):
        if self.initialized:
            raise RuntimeError('Cannot change param_strict after {} has been initialized'
                               .format(self.__class__.__name__))

        self._param_strict = value

    def initialize(self, module: Module, optimizer: Optimizer):
        super(GradualKSModifier, self).initialize(module, optimizer)
        layers = get_terminal_layers(module) if self._layers == ALL_TOKEN else \
            [get_layer(name, module) for name in self._layers]

        for layer in layers:
            found = False

            for param_name, par in layer.named_parameters():
                if param_name == self._param:
                    self._module_masks.append(KSLayerParamMask(layer, self._param))
                    found = True
                    break

            if self._param_strict and self._layers != ALL_TOKEN and not found:
                raise ValueError('Could not find required param {} in layer {} for {}'
                                 .format(self._param, layer, self.__class__.__name__))

    def update(self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int):
        if self.start_pending(epoch, steps_per_epoch):
            for mask in self._module_masks:
                mask.enabled = True

        if self.end_pending(epoch, steps_per_epoch) and not self._leave_enabled:
            for mask in self._module_masks:
                mask.enabled = False

        # set the mask tensors according to the new sparsity
        sparsity = interpolate(epoch, self.start_epoch, self.end_epoch,
                               self._init_sparsity, self._final_sparsity, self._inter_func)

        for mask in self._module_masks:
            mask.set_param_mask_from_sparsity(sparsity)

    def optimizer_post_step(self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int):
        # be sure to apply mask again after optimizer update because weights may have changed
        # (optimizer with momentum, not masking gradient)
        for mask in self._module_masks:
            mask.apply()


yaml.add_constructor(GradualKSModifier.YAML_KEY, GradualKSModifier.yaml_constructor)
yaml.add_constructor(GradualKSModifier.YAML_KEY, GradualKSModifier.yaml_constructor, yaml.SafeLoader)
