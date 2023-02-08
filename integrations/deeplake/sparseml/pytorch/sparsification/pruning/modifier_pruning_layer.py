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
Support and modifiers for pruning entire layers from models
"""
import math
from typing import Dict, List, Optional, Tuple, Union

from torch.nn import Module
from torch.optim.optimizer import Optimizer

from sparseml.optim import BaseModifier
from sparseml.pytorch.nn import Identity
from sparseml.pytorch.optim.analyzer_pruning import ModulePruningAnalyzer
from sparseml.pytorch.sparsification.modifier import (
    ModifierProp,
    PyTorchModifierYAML,
    ScheduledModifier,
    ScheduledUpdateModifier,
)
from sparseml.pytorch.utils import get_layer, get_prunable_layers, replace_layer
from sparseml.pytorch.utils.logger import BaseLogger
from sparseml.sparsification import SparsificationTypes
from sparseml.utils import ALL_PRUNABLE_TOKEN, ALL_TOKEN, validate_str_iterable


__all__ = [
    "LayerPruningModifier",
]


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
    """

    def __init__(
        self,
        layers: Union[str, List[str]],
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
        update_frequency: float = -1.0,
    ):
        super().__init__(
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=-1.0,
            end_comparator=-1,
        )
        self._layers = validate_str_iterable(
            layers, "{} for layers".format(self.__class__.__name__)
        )
        self._layer_modules = {}  # type: Dict[str, Module]
        self._layers_replaced = False
        self._last_logged_layers_replaced = None

    @BaseModifier.sparsification_types.getter
    def sparsification_types(self) -> List[SparsificationTypes]:
        """
        :return: the sparsification types this modifier instance will apply
        """
        return [SparsificationTypes.pruning, SparsificationTypes.structured]

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
        self._last_logged_layers_replaced = None

    @ScheduledModifier.log_call
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
        self,
        module: Module,
        optimizer: Optimizer,
        epoch: float,
        steps_per_epoch: int,
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

        if self._last_logged_layers_replaced != self._layers_replaced:
            self._last_logged_layers_replaced = self._layers_replaced
            layer_sparsities = [
                (name, 1 if self._layers_replaced else 0)
                for name in self._layer_modules.keys()
            ]
            for layer_sparsity in layer_sparsities:
                if isinstance(layer_sparsity, ModulePruningAnalyzer):
                    layer_sparsity = (
                        layer_sparsity.tag,
                        layer_sparsity.param_sparsity.item(),
                    )
                    self.log_scalar(
                        tag=f"LayerPruning/{layer_sparsity[0]}",
                        value=layer_sparsity[1],
                        epoch=epoch,
                        steps_per_epoch=steps_per_epoch,
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
