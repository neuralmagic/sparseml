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

from torch.nn import Module
from torch.optim.optimizer import Optimizer

from sparseml.pytorch.sparsification.modifier import (
    PyTorchModifierYAML,
    ScheduledUpdateModifier,
)
from sparseml.pytorch.utils import log_module_sparsification_info


__all__ = ["SparsificationLoggingModifier"]


@PyTorchModifierYAML()
class SparsificationLoggingModifier(ScheduledUpdateModifier):
    """
    Modifier to log the sparsification information of a module.
    Whenever this modifier is called, it will log the sparsification information
    of the module that it is attached to, using the logger(s) provided to it.

    | Sample yaml:
    |   !SparsificationLoggingModifier
    |       start_epoch: 0.0
    |       end_epoch: 10.0
    |       update_frequency: 1


    :param start_epoch: The epoch to start the modifier at
        (set to -1.0, so it starts immediately)
    :param end_epoch: The epoch to end the modifier at,
        (set to -1.0, so it doesn't end)
    :param update_frequency: if set to -1.0, will log module's
        sparsification information on each training step.
        If set to a positive integer, will update at the given frequency,
        at every epoch
    """

    def __init__(
        self,
        start_epoch: float,
        end_epoch: float = -1.0,
        update_frequency: float = 1.0,
    ):
        super(SparsificationLoggingModifier, self).__init__(
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            end_comparator=-1,
        )

    def update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Calls into the lr scheduler to step given the epoch
        Additionally will first set the lr to the init_lr if not set yet

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().update(module, optimizer, epoch, steps_per_epoch)
        log_module_sparsification_info(module=module, logger=self.loggers, step=epoch)
