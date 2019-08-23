from typing import Union, List, Tuple
import yaml
import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as TF
from torch.optim.optimizer import Optimizer

from ..modifier import ScheduledModifier, ALL_TOKEN
from ..utils import validate_str_list, get_terminal_layers, get_layer
from .tracker import ASLayerTracker


__all__ = ['ASRegModifier', 'REG_FUNCTIONS', 'REG_TENSORS']


REG_FUNCTIONS = ['l1', 'l2', 'relu']
REG_TENSORS = ['inp', 'out']


class ASRegModifier(ScheduledModifier):
    YAML_KEY = u'!ASRegModifier'

    @staticmethod
    def yaml_constructor(loader, node):
        instance = ASRegModifier.__new__(ASRegModifier)
        yield instance
        state = loader.construct_mapping(node, deep=True)
        instance.__init__(**state)

    def __init__(self, layers: Union[str, List[str]], alpha: float, reg_func: str = 'l1', reg_tens: str = 'inp',
                 start_epoch: float = -1.0, end_epoch: float = -1.0):
        super().__init__(start_epoch, end_epoch)

        self._layers = validate_str_list(layers, 'layers', ASRegModifier.YAML_KEY)
        self._alpha = alpha
        self._reg_func = reg_func
        self._reg_tens = reg_tens
        self._trackers = []  # type: List[ASLayerTracker]

        if not isinstance(self._alpha, float):
            raise Exception('alpha must be of float type for {}'.format(self.__class__.__name__))

        if self._reg_func not in REG_FUNCTIONS:
            raise Exception('{} is not a supported reg_func, available are {} for {}'
                            .format(self._reg_func, REG_FUNCTIONS, self.__class__.__name__))

        if self._reg_tens not in REG_TENSORS:
            raise Exception('{} is not a supported reg_tens, available are {} for {}'
                            .format(self._reg_tens, REG_TENSORS, self.__class__.__name__))

    def __del__(self):
        self._trackers.clear()

    @property
    def layers(self) -> Union[str, List[str]]:
        return self._layers

    @layers.setter
    def layers(self, value: Union[str, List[str]]):
        if self.initialized:
            raise Exception('Cannot change layers after {} has been initialized'.format(self.__class__.__name__))

        self._layers = value

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        self._alpha = value

    @property
    def reg_func(self) -> str:
        return self._reg_func

    @reg_func.setter
    def reg_func(self, value: str):
        self._reg_func = value

    @property
    def reg_tens(self) -> str:
        return self._reg_tens

    @reg_tens.setter
    def reg_tens(self, value: str):
        self._reg_tens = value

    def initialize(self, module: Module, optimizer: Optimizer):
        super(ASRegModifier, self).initialize(module, optimizer)
        layers = get_terminal_layers(module) if self._layers == ALL_TOKEN else \
            [get_layer(name, module) for name in self._layers]

        for layer in layers:
            self._trackers.append(ASLayerTracker(
                layer, track_input=self._reg_tens == 'inp', track_output=self._reg_tens == 'out',
                input_func=self._regularize_tracked, output_func=self._regularize_tracked
            ))

    def update(self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int):
        if self.start_pending(epoch, steps_per_epoch):
            for tracker in self._trackers:
                tracker.enable()

        if self.end_pending(epoch, steps_per_epoch):
            for tracker in self._trackers:
                tracker.disable()

    def loss_update(self, loss: Tensor, module: Module, optimizer: Optimizer,
                    epoch: float, steps_per_epoch: int) -> Tensor:
        if not self.enabled:
            return super().loss_update(loss, module, optimizer, epoch, steps_per_epoch)

        act_reg = 0.0

        for tracker in self._trackers:
            if self._reg_tens == 'inp':
                act_reg += tracker.tracked_input
            elif self._reg_tens == 'out':
                act_reg += tracker.tracked_output
            else:
                raise Exception('unsupported reg_tens given of {}'.format(self._reg_tens))

        # normalize across the number of trackers we have
        act_reg = act_reg / len(self._trackers)

        return loss + self._alpha * act_reg

    def _regularize_tracked(self, tens: Union[Tuple[Tensor, ...], Tensor]):
        if isinstance(tens, Tensor):
            tens = (tens,)

        reduced = torch.tensor(0.0)

        for ten in tens:
            if self._reg_func == 'l1':
                tens_reduced = ten.abs().sum()
            elif self._reg_func == 'l2':
                tens_reduced = ten.pow(2).sum()
            elif self._reg_func == 'relu':
                tens_reduced = TF.relu(reduced).sum()
            else:
                raise Exception('unsupported reg_func given of {}'.format(self._reg_func))

            reduced += tens_reduced / ten.numel()

        # normalize across all the tensors that were inputs or outputs for the layer
        reduced = reduced / len(tens)

        return reduced


yaml.add_constructor(ASRegModifier.YAML_KEY, ASRegModifier.yaml_constructor)
yaml.add_constructor(ASRegModifier.YAML_KEY, ASRegModifier.yaml_constructor, yaml.SafeLoader)
