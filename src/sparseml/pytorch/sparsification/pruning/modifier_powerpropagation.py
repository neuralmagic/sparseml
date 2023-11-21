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
from typing import List, Optional, Union

import torch
from torch.nn import Conv2d, Linear, Module, Parameter
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer

from sparseml.optim import ModifierProp
from sparseml.pytorch.sparsification.modifier import (
    PyTorchModifierYAML,
    ScheduledModifier,
)
from sparseml.pytorch.utils import (
    NamedLayerParam,
    get_named_layers_and_params_by_regex,
    get_prunable_layers,
)
from sparseml.pytorch.utils.logger import LoggerManager
from sparseml.utils import ALL_PRUNABLE_TOKEN, ALL_TOKEN, validate_str_iterable


__all__ = [
    "PowerpropagationModifier",
    "PowerpropagationWrapper",
]


_LOGGER = logging.getLogger(__name__)


class PowerpropagationWrapper(Module):
    def __init__(self, layer: Module, alpha: float = 1.0):
        super(PowerpropagationWrapper, self).__init__()

        if not isinstance(layer, Conv2d) and not isinstance(layer, Linear):
            raise ValueError("Powerpropagation only works with Linear and Conv layers")

        self.layer = layer
        # First set alpha to 1, then update it to the correct
        # value. This avoids replicating the code that updates
        # the layer weights.
        self.register_buffer("alpha", torch.tensor(1.0, requires_grad=False))
        self.set_alpha(alpha)

    def forward(self, x):
        weight = self.layer.weight * pow(abs(self.layer.weight), self.alpha - 1)

        if isinstance(self.layer, Conv2d):
            return F.conv2d(
                x,
                weight,
                self.layer.bias,
                self.layer.stride,
                self.layer.padding,
                self.layer.dilation,
                self.layer.groups,
            )
        elif isinstance(self.layer, Linear):
            return F.linear(x, weight, self.layer.bias)
        else:
            raise ValueError(
                "Powerpropagation only works with Linear and Conv2d layers"
            )

    def set_alpha(self, new_alpha):
        with torch.no_grad():

            self.layer.weight *= pow(abs(self.layer.weight), self.alpha / new_alpha - 1)
            # If there were any zeros in the weights, these may now be nan,
            # depending on the old and new values of alpha.
            self.layer.weight.data = torch.nan_to_num(self.layer.weight)
            self.alpha = torch.tensor(float(new_alpha))


@PyTorchModifierYAML()
class PowerpropagationModifier(ScheduledModifier):
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
        super(PowerpropagationModifier, self).__init__(
            start_epoch=start_epoch, end_epoch=end_epoch, end_comparator=-1
        )

        self._alpha = alpha
        self._strict = strict
        self._params = validate_str_iterable(
            params, "{} for params".format(self.__class__.__name__)
        )
        self._propagated_layers = {}

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
        :prams value: alpha (the power to which weights are raised during the
                      forward pass)
        """
        self._alpha = value

    @ModifierProp()
    def params(self) -> Union[str, List[str], None]:
        """
        :return: A list of full parameter names or regex patterns of names to apply
            pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
            will match to all parameters. __ALL_PRUNABLE__ will match to all ConvNd
            and Linear layers' weights
        """
        return self._params

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
        print(module.state_dict().keys())
        for name, layer, param in self._powerpropagated_layers:
            self._enable_powerprop(module, name, layer, param)
        print("\n\n\n", module.state_dict().keys())
        self._powerpropagation_enabled = True

    def _disable_module_powerpropagation(self, module: Module):
        if not self._powerpropagation_enabled:
            return
        for name, layer in self._propagated_layers.items():
            self._undo_enable_powerprop(module, name, layer)
        print("\n\n\n", module.state_dict().keys())
        self._powerpropagation_enabled = False

    # from https://pytorch.org/docs/stable/_modules/torch/ao/quantization/fuse_modules.html#fuse_modules  # noqa: E501
    # Generalization of setattr
    def _set_module(self, model, submodule_key, module):
        tokens = submodule_key.split(".")
        sub_tokens = tokens[:-1]
        cur_mod = model
        for s in sub_tokens:
            cur_mod = getattr(cur_mod, s)

        setattr(cur_mod, tokens[-1], module)

    def _enable_powerprop(
        self, model: Module, name: str, layer: Module, param: Parameter
    ):
        if isinstance(layer, Conv2d) or isinstance(layer, Linear):
            powerpropagated_layer = PowerpropagationWrapper(layer, self._alpha)
            if param.is_cuda:
                powerpropagated_layer = powerpropagated_layer.to(
                    torch.get_device(param)
                )
            self._propagated_layers[name] = powerpropagated_layer
            self._set_module(model, name, powerpropagated_layer)
        else:
            raise RuntimeError(f"don't know how do do powerpropagation for {layer}")
        return

    def _undo_enable_powerprop(self, model: Module, name: str, layer: Module):
        if isinstance(layer, PowerpropagationWrapper):
            # Setting alpha to 1 automatically updates the inner layer
            # weights to the correct non-exponentiated values.
            layer.set_alpha(1)
            self._set_module(model, name, layer.layer)
        else:
            raise RuntimeError(f"don't know how to undo powerpropagation for {layer}")
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

        _log(
            tag="PowerpropagationModifier/alpha",
            value=self._alpha,
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

        chosen = get_named_layers_and_params_by_regex(
            module,
            param_names,
            params_strict=self._strict,
        )
        return [(x[0], x[1], x[3]) for x in chosen]

    def _check_params_match(self, token: Union[str, List[str]]):
        if isinstance(token, str):
            return token in self._params or token == self._params

        if isinstance(self._params, str):
            return self._params in token

        return len(set(token).intersection(set(self._params))) > 0
