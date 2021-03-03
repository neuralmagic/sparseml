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
Modifier for changing parameters for regularization
"""


from typing import List, Union

from torch.nn import Module
from torch.optim import Optimizer

from sparseml.optim import ModifierProp
from sparseml.pytorch.optim.modifier import PyTorchModifierYAML, ScheduledModifier
from sparseml.pytorch.utils import PyTorchLogger
from sparseml.utils import ALL_TOKEN, convert_to_bool


__all__ = ["SetWeightDecayModifier"]


def _log_weight_decay(
    value: float, loggers: List[PyTorchLogger], epoch: float, steps_per_epoch: int
):
    step = round(epoch) if steps_per_epoch <= 0 else round(epoch * steps_per_epoch)
    for logger in loggers:
        logger.log_scalar("Modifier Weight Decay", value, step)


@PyTorchModifierYAML()
class SetWeightDecayModifier(ScheduledModifier):
    """
    Modifies the weight decay (L2 penalty) applied to with an optimizer during training

    | Sample yaml:
    |   !SetWeightDecayModifier
    |       start_epoch: 0.0
    |       weight_decay: 0.0
    |       param_groups: [0]
    |       log_types: __ALL__

    :param weight_decay: weight decay (L2 penalty) value to set for the given optimizer
    :param start_epoch: The epoch to start the modifier at
    :param param_groups: The indices of param groups in the optimizer to be modified.
        If None, all param groups will be modified. Default is None
    :param end_epoch: unused and should not be set
    :param log_types: The loggers to allow the learning rate to be logged to,
        default is __ALL__
    :param constant_logging: True to constantly log on every step,
        False to only log on an LR change and min once per epoch, default False
    """

    def __init__(
        self,
        weight_decay: float,
        start_epoch: float = -1.0,
        param_groups: Union[List[int], None] = None,
        end_epoch: float = -1.0,
        log_types: Union[str, List[str]] = ALL_TOKEN,
        constant_logging: bool = False,
    ):
        super().__init__(
            start_epoch=start_epoch,
            end_epoch=-1,
            log_types=log_types,
            end_comparator=-1,
        )

        self._weight_decay = weight_decay
        self._param_groups = param_groups
        self._constant_logging = convert_to_bool(constant_logging)
        self._update_since_last_log = False

    @ModifierProp()
    def weight_decay(self) -> float:
        """
        :return: weight decay (L2 penalty) value to set for the given optimizer
        """
        return self._weight_decay

    @weight_decay.setter
    def weight_decay(self, value: float):
        """
        :params value: weight decay (L2 penalty) value to set for the given optimizer
        """
        self._weight_decay = value

    @ModifierProp()
    def param_groups(self) -> Union[List[int], None]:
        """
        :return: The indices of param groups in the optimizer to be modified.
        If None, all param groups will be modified.
        """
        return self._param_groups

    @param_groups.setter
    def param_groups(self, value: Union[List[int], None]):
        """
        :params value: The indices of param groups in the optimizer to be modified.
        If None, all param groups will be modified.
        """
        self._param_groups = value

    @ModifierProp()
    def constant_logging(self) -> bool:
        """
        :return: True to constantly log on every step,
            False to only log on an LR change, default True
        """
        return self._constant_logging

    @constant_logging.setter
    def constant_logging(self, value: bool):
        """
        :param value: True to constantly log on every step,
            False to only log on an LR change, default True
        """
        self._constant_logging = value

    def update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        If start_pending(), updates the optimizers weight decay according to the
        parameters of this modifier

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().update(module, optimizer, epoch, steps_per_epoch)

        param_groups = (
            optimizer.param_groups
            if not self._param_groups
            else [optimizer.param_groups[idx] for idx in self._param_groups]
        )
        if self.start_pending(epoch, steps_per_epoch):
            for param_group in param_groups:
                param_group["weight_decay"] = self._weight_decay
        self._update_since_last_log = True

    def log_update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Check whether to log an update for the weight decay of the modifier
        If constant logging is enabled, then will always log
        Otherwise only logs after this modifier makes a change to the weight decay

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().log_update(module, optimizer, epoch, steps_per_epoch)

        if self._constant_logging or self._update_since_last_log:
            sample_param_group = optimizer.param_groups[
                self._param_groups[0] if self._param_groups else 0
            ]
            current_weight_decay = sample_param_group["weight_decay"]
            _log_weight_decay(
                current_weight_decay, self.loggers, epoch, steps_per_epoch
            )
            self._update_since_last_log = False
