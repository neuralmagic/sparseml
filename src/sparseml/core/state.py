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

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from sparseml.core.data import ModifiableData
from sparseml.core.event import Event
from sparseml.core.framework import Framework
from sparseml.core.logger import BaseLogger, LoggerManager
from sparseml.core.model import ModifiableModel
from sparseml.core.optimizer import ModifiableOptimizer


__all__ = ["State", "Data", "Hardware", "ModifiedState"]


@dataclass
class Data:
    """
    A dataclass to hold different data sets for training, validation,
    testing, and/or calibration. Each data set is a ModifiableData instance.

    :param train: The training data set
    :param val: The validation data set
    :param test: The testing data set
    :param calib: The calibration data set
    """

    train: Optional[ModifiableData] = None
    val: Optional[ModifiableData] = None
    test: Optional[ModifiableData] = None
    calib: Optional[ModifiableData] = None

    def reset(self):
        """
        Reset self to initial state
        """
        attribs = Data().__dict__
        for attrib_name, attrib_value in attribs.items():
            setattr(self, attrib_name, attrib_value)


@dataclass
class Hardware:
    """
    A dataclass to hold information about the hardware being used

    :param device: The current device being used for training
    :param devices: List of all devices to be used for training
    :param rank: The rank of the current device
    :param world_size: The total number of devices being used
    :param local_rank: The local rank of the current device
    :param local_world_size: The total number of devices being used on the local machine
    :param distributed: Whether or not distributed training is being used
    :param distributed_strategy: The distributed strategy being used
    """

    device: Optional[str] = None
    devices: Optional[List[str]] = None
    rank: Optional[int] = None
    world_size: Optional[int] = None
    local_rank: Optional[int] = None
    local_world_size: Optional[int] = None
    distributed: Optional[bool] = None
    distributed_strategy: Optional[str] = None


@dataclass
class State:
    """
    State class holds information about the current sparsification state

    :param framework: The framework being used
    :param model: The model being used for training
    :param teacher_model: The teacher model being used for training
    :param optimizer: The optimizer being used for training
    :param optim_wrapped: Whether or not the optimizer has been wrapped
    :param loss: The loss function being used for training
    :param batch_data: The current batch of data being used for training
    :param data: The data sets being used for training, validation, testing,
        and/or calibration, wrapped in a Data instance
    :param hardware: Hardware Instance holding info about the target hardware being used
    :param start_event: The start event to begin training
    :param last_event: The last event to stop training
    :param loggers: LoggerManager instance holding all the loggers to log
    :param model_log_cadence: The cadence to log model information w.r.t epochs.
        If 1, logs every epoch. If 2, logs every other epoch, etc. Default is 1.
    """

    framework: Framework
    model: ModifiableModel = None
    teacher_model: ModifiableModel = None
    optimizer: ModifiableOptimizer = None
    optim_wrapped: bool = None
    loss: Any = None
    batch_data: Any = None
    data = Data()
    hardware = Hardware()
    start_event: Event = None
    last_event: Event = None
    loggers: Optional[LoggerManager] = None
    model_log_cadence: Optional[float] = None
    _last_log_step: Union[float, int, None] = None

    @property
    def sparsification_ready(self) -> bool:
        return (
            self.model is not None
            and self.optimizer is not None
            # and self.loss is not None
            # and self.batch_data is not None
        )

    def update(
        self,
        model: Any = None,
        teacher_model: Any = None,
        optimizer: Any = None,
        attach_optim_callbacks: bool = True,
        train_data: Any = None,
        val_data: Any = None,
        test_data: Any = None,
        calib_data: Any = None,
        copy_data: bool = True,
        start: float = None,
        steps_per_epoch: int = None,
        batches_per_step: int = None,
        loggers: Union[None, LoggerManager, List[BaseLogger]] = None,
        model_log_cadence: Optional[float] = None,
        **kwargs,
    ) -> Dict:
        """
        Update the state with the given parameters

        :param model: The model to update the state with
        :param teacher_model: The teacher model to update the state with
        :param optimizer: The optimizer to update the state with
        :param attach_optim_callbacks: Whether or not to attach optimizer callbacks
        :param train_data: The training data to update the state with
        :param val_data: The validation data to update the state with
        :param test_data: The testing data to update the state with
        :param calib_data: The calibration data to update the state with
        :param copy_data: Whether or not to copy the data
        :param start: The start index to update the state with
        :param steps_per_epoch: The steps per epoch to update the state with
        :param batches_per_step: The batches per step to update the state with
        :param loggers: the logger manager to setup logging important info and
            milestones to, also accepts a list of BaseLogger(s)
        :param model_log_cadence: The cadence to log model information w.r.t epochs.
            If 1, logs every epoch. If 2, logs every other epoch, etc. Default is 1.
        :param kwargs: Additional keyword arguments to update the state with
        """
        if model is not None:
            self.model = ModifiableModel(framework=self.framework, model=model)
        if teacher_model is not None:
            self.teacher_model = ModifiableModel(
                framework=self.framework, model=teacher_model
            )
        if optimizer is not None:
            self.optim_wrapped = attach_optim_callbacks
            self.optimizer = ModifiableOptimizer(
                framework=self.framework, optimizer=optimizer
            )

        if train_data is not None:
            self.data.train = train_data if not copy_data else deepcopy(train_data)
        if val_data is not None:
            self.data.val = val_data if not copy_data else deepcopy(val_data)
        if test_data is not None:
            self.data.test = test_data if not copy_data else deepcopy(test_data)
        if calib_data is not None:
            self.data.calib = calib_data if not copy_data else deepcopy(calib_data)

        if "device" in kwargs:
            self.hardware.device = kwargs["device"]

        if (
            start is not None
            or steps_per_epoch is not None
            or batches_per_step is not None
        ):
            if self.start_event is None:
                self.start_event = Event()

            if start is not None:
                self.start_event.current_index = start
            if steps_per_epoch is not None:
                self.start_event.steps_per_epoch = steps_per_epoch
            if batches_per_step is not None:
                self.start_event.batches_per_step = batches_per_step

        loggers = loggers or []
        if isinstance(loggers, List):
            loggers = LoggerManager(loggers)
        self.loggers = loggers

        if model_log_cadence is not None:
            self.model_log_cadence = model_log_cadence
        return kwargs


@dataclass
class ModifiedState:
    """
    A dataclass to represent a modified model,
    optimizer, and loss

    :param model: The modified model
    :param optimizer: The modified optimizer
    :param loss: The modified loss
    :param modifier_data: The modifier data used to modify the
        model, optimizer, and loss
    """

    model: Optional[Any] = None
    optimizer: Optional[Any] = None
    loss: Optional[Any] = None
    modifier_data: Optional[List[Dict[str, Any]]] = None

    def __init__(self, model, optimizer, loss, modifier_data):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.modifier_data = modifier_data
