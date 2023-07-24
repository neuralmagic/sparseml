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
Modifier for models through powerproagation.

"""


import logging
import math
import warnings
from itertools import cycle
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import torch
from torch.nn import Module, Conv2d, Linear, Parameter
from torch.optim.optimizer import Optimizer

from sparseml.optim import BaseModifier, ModifierProp
from sparseml.pytorch.sparsification.modifier import (
    PyTorchModifierYAML,
    ScheduledModifier,
    ScheduledUpdateModifier,
)
from sparseml.utils import (
    ALL_PRUNABLE_TOKEN,
    ALL_TOKEN,
    FROM_PARAM_TOKEN,
    interpolate,
    validate_str_iterable,
)

from sparseml.pytorch.utils import BaseLogger, NamedLayerParam, get_named_layers_and_params_by_regex, get_prunable_layers,  tensors_module_forward, tensors_to_device
from sparseml.sparsification import SparsificationTypes
from sparseml.pytorch.utils.logger import LoggerManager
from torch.nn import functional as F


__all__ = [
    "PowerpropagationModifier",
]


_LOGGER = logging.getLogger(__name__)



@PyTorchModifierYAML()
class PowerpropagationModifier(ScheduledUpdateModifier):
    """
    Does powerpropagation. TODO: more here.

    | Sample yaml:
    |   !PowerpropagationModifier
    |       start_epoch: 0.0
    |       end_epoch: 100
    |       alpha: 2.0
    |       params: __ALL_PRUNABLE__
    |       strict: True

    :param start_epoch: The epoch to start the modifier at
    :param alpha: The degree weights should be raised to before the standard forward
        pass, preserving the original sign of the weight. Noninteger weights are OK.
    :param params: A list of full parameter names or regex patterns of names to apply
        pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
        will match to all parameters. __ALL_PRUNABLE__ will match to all ConvNd
        and Linear layers' weights. If a sparsity to param mapping is defined by
        final_sparsity, then params should be set to []
    :param strict: if True, will raise an error if any module types or submodules in
        scheme_overrides or ignore are not found in a given module. Default True
    :param end_epoch: The epoch at which the architecture changes will be reversed,
        converting the network back to a normal architecture. Note that if this is not
        set, or if it is set for after the network finishes training, the architecture
        changes will become part of the model, making it largely incompatible with
        other frameworks.
    """

    def __init__(
        self,
        start_epoch: Union[int, float],
        end_epoch: Union[int, float],
        params: Union[str, List[str]],
        alpha: float = 1.0,
        strict: bool = True,
    ):
        super(PowerpropagationModifier, self).__init__(start_epoch=start_epoch,
               end_epoch=end_epoch, end_comparator=-1)

        self._alpha = alpha
        self._strict = strict
        self._params = validate_str_iterable(
            params, "{} for params".format(self.__class__.__name__)
        )

        self._validate_params()

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
        self._powerpropagated_layers = self._create_named_layers_and_params(module)



    @ModifierProp()
    def alpha(self) -> Optional[float]:
        """
        :return: alpha (the power to which weights are raised during the forward pass)
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        """
        :prams value: alpha (the power to which weights are raised during the forward pass)
        """
        self._alpha = value


    def update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        If start_pending(), converts layers to powerpropagated layers
        If end_pending(), undoes the conversion

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().update(module, optimizer, epoch, steps_per_epoch)
        self._check_powerpropagation_update(module, epoch, steps_per_epoch)

    def _check_powerpropagation_update(
        self, module: Module, epoch: float, steps_per_epoch: int
    ):
        if self.start_pending(epoch, steps_per_epoch):
            self._enable_module_powerpropagation(module)
        if self.end_pending(epoch, steps_per_epoch):
            self._disable_module_powerpropagation(module)

        # TODO: Make this do something useful
        self._log_powerpropagation(module, epoch, steps_per_epoch)


    def _enable_module_powerpropagation(self, module: Module):
        for name, layer, param in self._powerpropagated_layers:
            print(name)
            self._replace_with_powerprop(layer, param)
        self._powerpropagation_enabled = True

    def _disable_module_powerpropagation(self, module: Module):
        if not self._powerpropagation_enabled:
            return
        for name, layer, param in self._powerpropagated_layers:
            self._undo_replace_with_powerprop(layer, param)
        self._powerpropagation_enabled = False


    def _replace_with_powerprop(self, module: Module, param: Parameter):
        alpha = self._alpha
        def powerpropagated_convolution(self, input):
            powerpropagated_weight = self.weight *pow(abs(self.weight), alpha-1)
            return self._conv_forward(input, powerpropagated_weight, self.bias)
        def powerpropagated_linear(self, input):
            powerpropagated_weight = self.weight *pow(abs(self.weight), alpha-1)
            return F.linear(input, powerpropagated_weight, self.bias)
        if isinstance(module, Conv2d):
            bound_method = powerpropagated_convolution.__get__(module, module.__class__)
            setattr(module, 'forward', bound_method)
        elif isinstance(module, Linear):
            bound_method = powerpropagated_linear.__get__(module, module.__class__)
            setattr(module, 'forward', bound_method)
        else:
            raise RuntimeError(f"don't know how do do powerpropagation for {module.__class__()}")
        
        with torch.no_grad():
            param = param*pow(abs(param), 1/alpha - 1)
        return

    def _undo_replace_with_powerprop(self, module: Module, param: Parameter):
        def normal_convolution(self, input):
            return self._conv_forward(input, self.weight, self.bias)
        def normal_linear(self, input):
            return F.linear(input, self.weight, self.bias)
        if isinstance(module, Conv2d):
            bound_method = normal_convolution.__get__(module, module.__class__)
            setattr(module, 'forward', bound_method)
        elif isinstance(module, Linear):
            bound_method = normal_linear.__get__(module, module.__class__)
            setattr(module, 'forward', bound_method)
        else:
            raise RuntimeError(f"don't know how to undo powerpropagation for {module.__class__()}")
        
        with torch.no_grad():
            param = param*pow(abs(param), self._alpha - 1)
        return

    def _validate_params(self):
        self.validate_schedule()

    def _log_powerpropagation(
        self,
        module: Module,
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

        def _log(tag, value):
            self.log_scalar(
                tag=tag,
                value=value,
                epoch=epoch,
                steps_per_epoch=steps_per_epoch,
            )


        print("!!!!!!!!!!!!!!! did a fake log for now")
        _log(
            tag=f"PowerpropagationModifier/something",
            value=1337,
        )


    def _create_named_layers_and_params(self, module: Module) -> List[NamedLayerParam]:
        if self._check_params_match(ALL_TOKEN):
            param_names = ["re:.*"]
        elif self._check_params_match(ALL_PRUNABLE_TOKEN):
            param_names = [
                name + ".weight" for (name, _) in get_prunable_layers(module)
            ]
        else:
            param_names = self._params

        chosen =  get_named_layers_and_params_by_regex(
            module,
            param_names,
            params_strict=True,
        )
        return ([(x[0 ], x[1], x[3]) for x in chosen])


    def _check_params_match(self, token: Union[str, List[str]]):
        if isinstance(token, str):
            return token in self._params or token == self._params

        if isinstance(self._params, str):
            return self._params in token

        return len(set(token).intersection(set(self._params))) > 0
