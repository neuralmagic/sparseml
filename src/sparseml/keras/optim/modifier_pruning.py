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
Modifiers for inducing / enforcing kernel sparsity (model pruning)
on models while pruning.
"""

from typing import Dict, List, Union

import tensorflow

from sparseml.keras.optim.mask_pruning import (
    MaskedLayer,
    PruningScheduler,
    remove_pruning_masks,
)
from sparseml.keras.optim.modifier import (
    KerasModifierYAML,
    ScheduledModifier,
    ScheduledUpdateModifier,
)
from sparseml.keras.optim.utils import get_layer_name_from_param
from sparseml.keras.utils import KerasLogger, LoggerSettingCallback, keras
from sparseml.sparsification import (
    ConstantPruningModifier as BaseConstantPruningModifier,
)
from sparseml.sparsification import GMPruningModifier as BaseGMPruningModifier


__all__ = ["ConstantPruningModifier", "GMPruningModifier"]


class FunctionalScheduler(PruningScheduler):
    """
    Pruning scheduler based on a predefined function

    :param init_sparsity: initial sparsity
    :param final_sparsity: final sparsity
    :param start_step: the starting step
    :param end_step: the ending step
    :param update_frequency_steps: the number of frequency steps to update
    :param inter_func: function to update sparsity over time
    """

    def __init__(
        self,
        init_sparsity: float,
        final_sparsity: float,
        start_step: int,
        end_step: int,
        update_frequency_steps: int,
        inter_func: str = "cubic",
    ):
        self._init_sparsity = init_sparsity
        self._final_sparsity = final_sparsity
        self._start_step = start_step
        self._end_step = end_step
        self._update_frequency_steps = update_frequency_steps
        self._inter_func = inter_func

    @property
    def init_sparsity(self):
        return self._init_sparsity

    @property
    def final_sparsity(self):
        return self._final_sparsity

    @property
    def start_step(self):
        return self._start_step

    @property
    def end_step(self):
        return self._end_step

    @property
    def update_frequency_steps(self):
        return self._update_frequency_steps

    @property
    def inter_func(self):
        return self._inter_func

    @property
    def exponent(self) -> float:
        """
        :return: the exponent to be used in for the sparsity schedule
        """

        if self._inter_func == "linear":
            return 1.0

        if self._inter_func == "cubic":
            return 3.0

        if self._inter_func == "inverse_cubic":
            return 1 / 3.0

        raise ValueError(
            "unrecognized value given for inter_func of {}".format(self._inter_func)
        )

    def should_prune(self, step: int) -> bool:
        """
        Check if the given step is a right time for pruning

        :param step: training step
        :return: True if pruning should take place; False otherwise
        """
        sched_start = step == self.start_step
        sched_end = step == self.end_step
        sched_active = step > self.start_step and step < self.end_step
        sched_active_inclusive = sched_active or sched_start or sched_end

        if self.update_frequency_steps <= 0:
            sched_update = True
        else:
            sched_update = (step - self.start_step) % self.update_frequency_steps == 0
        sched_update_ready = sched_start or sched_end or sched_update
        update_ready = sched_active_inclusive and sched_update_ready
        return update_ready

    def target_sparsity(self, step: int, **kwargs):
        """
        Compute the target sparsity at the given step

        :param step: training step
        :param kwargs: optional keyword params that a specific scheduler might need
        :return: target sparsity
        """
        sched_before = step < self.start_step
        sched_start = step == self.start_step
        sched_active = step > self.start_step and step < self.end_step

        percentage = min(
            1.0, max(0.0, (step - self.start_step) / (self.end_step - self.start_step))
        )
        exp_percentage = 1 - pow(1 - percentage, self.exponent)
        calc_sparsity = (
            self._final_sparsity - self._init_sparsity
        ) * exp_percentage + self._init_sparsity

        if sched_before:
            sparsity = 0.0
        elif sched_start:
            sparsity = self._init_sparsity
        elif sched_active:
            sparsity = calc_sparsity
        else:
            sparsity = self._final_sparsity
        return sparsity

    def get_config(self):
        config = {
            "class_name": self.__class__.__name__,
            "config": {
                "init_sparsity": self.init_sparsity,
                "final_sparsity": self.final_sparsity,
                "start_step": self.start_step,
                "end_step": self.end_step,
                "update_frequency_steps": self.update_frequency_steps,
                "inter_func": self.inter_func,
            },
        }
        return config


class SparsityFreezer(PruningScheduler):
    """
    A sparsity scheduler that fix the sparsity level based on
    a given tensor over a period of time

    :param start_step: starting step to begin the schedule
    :param end_step: ending step to end the schedule
    """

    def __init__(
        self,
        start_step: int,
        end_step: int,
    ):
        self._start_step = start_step
        self._end_step = end_step

    @property
    def start_step(self):
        return self._start_step

    @property
    def end_step(self):
        return self._ends_step

    def should_prune(self, step: int) -> bool:
        """
        Check if the given step is a right time for pruning

        :param step: training step
        :return: True if pruning should take place; False otherwise
        """
        return step in [self._start_step, self._end_step]

    def target_sparsity(self, step: int, tensor=None) -> float:
        """
        Compute the target sparsity at the given step

        :param step: training step
        :param tensor: tensor (e.g., weight) to compute the sparsity
        :return: target sparsity, or None
        """
        if tensor is None:
            raise ValueError("Invalid empty tensor")
        if self._start_step <= step < self._end_step:
            mask = tensorflow.cast(tensorflow.not_equal(tensor, 0.0), tensor.dtype)
            sparsity = float(
                tensorflow.math.reduce_sum(1.0 - mask).numpy() / tensorflow.size(tensor)
            )
        elif step == self._end_step:
            sparsity = 0.0
        else:
            # Undefined sparsity
            sparsity = None
        return sparsity

    def get_config(self):
        config = {
            "class_name": self.__class__.__name__,
            "start_step": self.start_step,
            "end_step": self.end_step,
        }
        return config


class PruningModifierCallback(keras.callbacks.Callback):
    """
    A callback to update masks and weights at the end of certain training step

    :param prunable_layers: list of masked layers
    """

    def __init__(self, prunable_layers, optim_iters):
        self.prunable_layers = prunable_layers
        self.optim_iters = optim_iters
        self.step = None

    def on_train_begin(self, logs=None):
        """
        Called at the begin of training

        :param logs: dictionary of logs (see Keras Callback doc)
        """
        self.step = keras.backend.get_value(self.optim_iters)
        keras.backend.batch_set_value(
            [(layer.global_step, self.step) for layer in self.prunable_layers]
        )

    def on_train_batch_begin(self, batch, logs=None):
        """
        Called at the begin of a batch in training

        :param batch: batch index in current epoch
        :param logs: dictionary of logs (see Keras Callback doc)
        """
        keras.backend.batch_set_value(
            [(layer.global_step, self.step) for layer in self.prunable_layers]
        )

    def on_train_batch_end(self, batch, logs=None):
        """
        Called at the end of a batch in training

        :param batch: batch index in current epoch
        :param logs: dictionary of logs (see Keras Callback doc)
        """
        for layer in self.prunable_layers:
            layer.mask_updater.conditional_update(training=True)
        self.step = self.step + 1

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of a training epoch

        :param epoch: epoch index
        :param logs: dictionary of logs (see Keras Callback doc)
        """
        for layer in self.prunable_layers:
            layer.mask_updater.apply_masks()


class SparsityLoggingCallback(LoggerSettingCallback):
    """
    Callback to log sparsity level

    :param loggers: an instance of KerasLogger or a list of those instances
    :param prunable_layers: list of masked layers
    :param start_step: start step
    """

    def __init__(
        self,
        loggers: Union[KerasLogger, List[KerasLogger]],
        prunable_layers: List[MaskedLayer],
        start_step: int,
    ):
        super().__init__(loggers)
        self._prunable_layers = prunable_layers
        self._step = None
        self._start_step = start_step

    def on_train_begin(self, logs=None):
        """
        Called at the begin of training

        :param logs: dictionary of logs (see Keras Callback doc)
        """
        super().on_train_begin(logs)
        self._step = keras.backend.get_value(self._start_step)

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of a training epoch

        :param epoch: epoch index
        :param logs: dictionary of logs (see Keras Callback doc)
        """
        super().on_epoch_end(epoch, logs)
        for logger in self._loggers:
            if logger.update_freq == "epoch":
                logged_data = self._get_log_data()
                self._log(logger, logged_data)

    def on_train_batch_end(self, batch, logs=None):
        """
        Called at the end of a batch in training

        :param batch: batch index in current epoch
        :param logs: dictionary of logs (see Keras Callback doc)
        """
        super().on_train_batch_end(batch, logs)
        for logger in self._loggers:
            if logger.update_freq == "batch" or (
                isinstance(logger.update_freq, int)
                and self._step % logger.update_freq == 0
            ):
                logged_data = self._get_log_data()
                self._log(logger, logged_data)

        # Keep track of the step count
        self._step += 1

    def _get_log_data(self):
        """
        Add tensors in the summaries for tensorboard logging

        :return: a dictionary of named tensors
        """
        log_data = {}
        for layer in self._prunable_layers:
            for masked_param in layer.pruning_vars:
                sparsity = tensorflow.math.subtract(
                    1, tensorflow.math.reduce_mean(masked_param.mask)
                )
                log_data["sparsity@{}".format(masked_param.name)] = sparsity
        return log_data

    def _log(self, logger: KerasLogger, log_data: Dict):
        """
        Retrieve logging values from modifiers and add to Tensorboard
        """
        for name, value in log_data.items():
            logger.log_scalar(name, value, step=self._step)


@KerasModifierYAML()
class ConstantPruningModifier(BaseConstantPruningModifier, ScheduledModifier):
    """
    Holds the sparsity level and shape for a given param constant while training.
    Useful for transfer learning use cases.

    | Sample yaml:
    |   !ConstantPruningModifier
    |       params: __ALL__
    |       start_epoch: 0.0
    |       end_epoch: 10.0

    :param params: List of str names or regex patterns of names for the parameter
        variables to apply the KS modifier to. Regex patterns must be specified
        with the prefix 're:'. Can also use the token __ALL__ to specify all
        prunable layers and weights
    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    """

    def __init__(
        self,
        params: Union[str, List[str]],
        start_epoch: float = -1,
        end_epoch: float = -1,
    ):
        super(ConstantPruningModifier, self).__init__(
            params=params,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            end_comparator=None,
        )
        self._layer_names = [get_layer_name_from_param(p) for p in self._params]
        self._masked_layers = []

        self._sparsity_scheduler = None
        self._mask_type = "unstructured"

    @property
    def layer_names(self) -> List[str]:
        return self._layer_names

    @property
    def update_ready(self):
        """
        :return: the created update_ready tensor for setting the pruning ops
            if create_ops has been called, else None
        """
        return self._update_ready

    @property
    def sparsity(self) -> Union[None, tensorflow.Tensor]:
        """
        :return: the created sparsity tensor for setting the pruning ops
            if create_ops has been called, else None
        """
        return self._sparsity

    def is_pruning_step(self, step: int, steps_per_epoch, tensor=None):
        begin_step, end_step = self.start_end_steps(steps_per_epoch, after_optim=False)
        is_start_step = step == begin_step
        is_end_step = step == end_step
        self._update_ready = is_start_step or is_end_step
        if is_start_step:
            if tensor is None:
                raise RuntimeError("Unexpected empty weight")
            mask = tensorflow.cast(tensorflow.not_equal(tensor, 0.0), tensor.dtype)
            self._sparsity = tensorflow.math.reduce_sum(
                1.0 - mask
            ).numpy() / tensorflow.size(tensor)
        elif is_end_step:
            mask = tensorflow.ones_like(tensor)
            self._sparsity = 0.0
        else:
            self._sparsity = None
            mask = None
        return self._update_ready, self._sparsity, mask

    def _create_sparsity_scheduler(self, steps_per_epoch):
        begin_step, end_step = self.start_end_steps(steps_per_epoch, after_optim=False)
        sparsity_scheduler = SparsityFreezer(begin_step, end_step)
        return sparsity_scheduler

    def _clone_layer(self, layer: keras.layers.Layer):
        cloned_layer = layer
        if layer.name in self.layer_names:  # TODO: handle regex params
            cloned_layer = MaskedLayer(
                layer, self._sparsity_scheduler, self._mask_type, name=layer.name
            )
            self._masked_layers.append(cloned_layer)
        return cloned_layer

    def modify(
        self,
        model,
        optimizer,
        steps_per_epoch: int,
        loggers: Union[KerasLogger, List[KerasLogger]] = None,
        input_tensors: tensorflow.Tensor = None,
    ):
        """
        Modify model and optimizer

        :param model: a model to be modified
        :param optimizer: an optimizer to be modified
        :param steps_per_epoch: number of steps per epoch
        :param loggers: list of loggers
        :param input_tensors: optional input tensors
        :return: modified model, optimizer and callbacks
        """
        model, optimizer, callback = super(ConstantPruningModifier, self).modify(
            model,
            optimizer,
            steps_per_epoch,
            loggers=loggers,
            input_tensors=input_tensors,
        )
        self._sparsity_scheduler = self._create_sparsity_scheduler(steps_per_epoch)
        cloned_model = keras.models.clone_model(
            model,
            input_tensors,
            clone_function=self._clone_layer,
        )
        pruning_step_callback = PruningModifierCallback(self._masked_layers)
        callbacks = [pruning_step_callback]
        if loggers is not None:
            sparsity_logging_callback = SparsityLoggingCallback(
                loggers, self._masked_layers, optimizer.iterations
            )
            callbacks.append(sparsity_logging_callback)
        return cloned_model, optimizer, callbacks

    def finalize(self, model: keras.Model):
        """
        Remove extra information related to the modifier from the model that is
        not necessary for exporting

        :param model: a Keras model
        :return: a new Keras model
        """
        return remove_pruning_masks(model)


@KerasModifierYAML()
class GMPruningModifier(BaseGMPruningModifier, ScheduledUpdateModifier):
    """
    Gradually applies kernel sparsity to a given variable or variables from
    init_sparsity until final_sparsity is reached over a given amount of time and
    applied with an interpolated function for each step taken.

    Applies based on magnitude pruning without any structure to the pruning.

    | Sample yaml:
    |   !GMPruningModifier
    |       params: __ALL__
    |       init_sparsity: 0.05
    |       final_sparsity: 0.8
    |       start_epoch: 0.0
    |       end_epoch: 10.0
    |       update_frequency: 1.0
    |       inter_func: cubic
    |       mask_type: unstructured
    |       leave_enabled: True

    :param params: List of str names or name regex patterns for the variables in the
        graph to apply the KS modifier to.  Regex patterns must be specified with
        the prefix 're:'.  __ALL__ will match to all parameters.
    :param init_sparsity: The initial sparsity for the variable to
        start with at start_epoch
    :param final_sparsity: The final sparsity for the variable to end with at end_epoch
    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param update_frequency: The number of epochs or fraction of epochs to
        update at between start and end
    :param leave_enabled: True to continue masking the weights after end_epoch,
        False to stop masking. Should be set to False if exporting the result
        immediately after or doing some other prune
    :param inter_func: The type of interpolation function to use:
        [linear, cubic, inverse_cubic]
    :param mask_type: String to define type of sparsity (options: ['unstructured',
        'channel', 'filter']), List to define block shape of a parameter's in and out
        channels. default is 'unstructured'
    :param leave_enabled: True to continue masking the weights after end_epoch,
        False to stop masking. Should be set to False if exporting the result
        immediately after or doing some other prune
    """

    def __init__(
        self,
        params: Union[str, List[str]],
        init_sparsity: float,
        final_sparsity: float,
        start_epoch: float,
        end_epoch: float,
        update_frequency: float,
        inter_func: str = "cubic",
        mask_type: Union[str, List[int]] = "unstructured",
        leave_enabled: bool = True,
    ):
        super(GMPruningModifier, self).__init__(
            params=params,
            init_sparsity=init_sparsity,
            final_sparsity=final_sparsity,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            inter_func=inter_func,
            mask_type=mask_type,
            leave_enabled=leave_enabled,
            min_start=-1.0,
            min_end=0.0,
            end_comparator=1,
            min_frequency=-1.0,
        )
        self._layer_names = [get_layer_name_from_param(p) for p in self._params]
        self._prune_op_vars = None
        self._update_ready = None
        self._sparsity = None
        self._mask_initializer = None

        self._masked_layers = []

    @property
    def layer_names(self) -> List[str]:
        return self._layer_names

    @property
    def update_ready(self):
        """
        :return: the created update_ready tensor for setting the pruning ops
            if create_ops has been called, else None
        """
        return self._update_ready

    @property
    def sparsity(self) -> Union[None, tensorflow.Tensor]:
        """
        :return: the created sparsity tensor for setting the pruning ops
            if create_ops has been called, else None
        """
        return self._sparsity

    def _create_sparsity_scheduler(self, steps_per_epoch):
        begin_step, end_step = self.start_end_steps(steps_per_epoch, after_optim=False)
        update_frequency_steps = self.update_frequency_steps(steps_per_epoch)
        sparsity_scheduler = FunctionalScheduler(
            self.init_sparsity,
            self.final_sparsity,
            begin_step,
            end_step,
            update_frequency_steps,
            self.inter_func,
        )
        return sparsity_scheduler

    def _clone_layer(self, layer: keras.layers.Layer):
        cloned_layer = layer
        if (
            layer.name in self.layer_names
        ):  # TODO: handle regex params --- see create_ops in TF version
            cloned_layer = MaskedLayer(
                layer, self._sparsity_scheduler, self._mask_type, name=layer.name
            )
            self._masked_layers.append(cloned_layer)
        return cloned_layer

    def modify(
        self,
        model,
        optimizer,
        steps_per_epoch: int,
        loggers: Union[KerasLogger, List[KerasLogger]] = None,
        input_tensors: tensorflow.Tensor = None,
    ):
        """
        Modify model and optimizer, and provide callbacks to process the model

        :param model: a model to be modified with prunable layers wrapped by masks
        :param optimizer: an optimizer to be modified; in this case, no change to it
        :param steps_per_epoch: number of steps per epoch
        :param loggers: list of loggers
        :param input_tensors: optional input tensors
        :return: modified model, optimizer and callbacks
        """
        # TODO: incorporate the returned callback into the final callbacks to return
        model, optimizer, callback = super(GMPruningModifier, self).modify(
            model,
            optimizer,
            steps_per_epoch,
            loggers=loggers,
            input_tensors=input_tensors,
        )

        self._sparsity_scheduler = self._create_sparsity_scheduler(steps_per_epoch)

        # Clone model and additional set up
        cloned_model = keras.models.clone_model(
            model,
            input_tensors,
            clone_function=self._clone_layer,
        )

        # Pruning step call back and additional set up
        pruning_step_callback = PruningModifierCallback(
            self._masked_layers, optimizer.iterations
        )
        callbacks = [pruning_step_callback]
        if loggers is not None:
            sparsity_logging_callback = SparsityLoggingCallback(
                loggers, self._masked_layers, optimizer.iterations
            )
            callbacks.append(sparsity_logging_callback)
        return cloned_model, optimizer, callbacks

    @property
    def prunable_layers(self):
        return self._masked_layers

    def finalize(self, model: keras.Model):
        """
        Remove extra information related to the modifier from the model that is
        not necessary for exporting

        :param model: a Keras model
        :return: a new Keras model
        """
        return remove_pruning_masks(model)
