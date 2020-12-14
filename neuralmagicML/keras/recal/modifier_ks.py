"""
Modifiers for inducing / enforcing kernel sparsity (model pruning)
on models while pruning.
"""

from typing import Union, List, Tuple, Dict, Any
import hashlib

import tensorflow as tf

from neuralmagicML.utils import (
    ALL_TOKEN,
    validate_str_iterable,
    convert_to_bool,
)
from neuralmagicML.keras.recal.modifier import (
    EXTRAS_KEY_SUMMARIES,
    ModifierProp,
    KerasModifierYAML,
    ScheduledModifier,
    ScheduledUpdateModifier,
)

from neuralmagicML.keras.recal.sparsity_mask import (
    SparsityMaskCreator,
    load_mask_creator,
)

from neuralmagicML.keras.recal.masked_layer import MaskedLayer, PruningScheduler

from neuralmagicML.keras.recal.utils import get_layer_name_from_param


__all__ = ["ConstantKSModifier", "GradualKSModifier"]


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
    def start_step(self):
        return self._start_step

    @property
    def end_step(self):
        return self._end_step

    @property
    def update_frequency_steps(self):
        return self._update_frequency_steps

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
        sched_before = step < self.start_step
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
        sched_end = step == self.end_step
        sched_active = step > self.start_step and step < self.end_step
        sched_active_inclusive = sched_active or sched_start or sched_end

        if self.update_frequency_steps <= 0:
            sched_update = True
        else:
            sched_update = (step - self.start_step) % self.update_frequency_steps == 0

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
        self._sparsity = None

    def should_prune(self, step: int) -> bool:
        """
        Check if the given step is a right time for pruning

        :param step: training step
        :return: True if pruning should take place; False otherwise
        """
        return step in [self._start_step, self._end_step]

    def _get_sparsity(self, tensor: tf.Tensor):
        if self._sparsity is None:
            mask = tf.cast(tf.not_equal(tensor, 0.0), tensor.dtype)
            self._sparsity = tf.math.reduce_sum(1.0 - mask).numpy() / tf.size(tensor)
        return self._sparsity

    def target_sparsity(self, step: int, tensor=None) -> float:
        """
        Compute the target sparsity at the given step

        :param step: training step
        :param tensor: tensor (e.g., weight) to compute the sparsity
        :return: target sparsity
        """
        if tensor is None:
            raise ValueError("Invalid empty tensor")
        if self._start_step <= step < self._end_step:
            sparsity = self._get_sparsity(step)
        elif step == self._end_step:
            sparsity = 0.0
        else:
            # Undefined sparsity
            sparsity = None
        return sparsity


class KSModifierCallback(tf.keras.callbacks.Callback):
    """
    A callback to update masks and weights at the end of certain training step

    :param prunable_layers: list of masked layers
    """

    def __init__(self, prunable_layers):
        self.prunable_layers = prunable_layers
        self.step = None

    def on_train_begin(self, logs=None):
        self.step = tf.keras.backend.get_value(self.model.optimizer.iterations)
        tf.keras.backend.batch_set_value(
            [(layer.global_step, self.step) for layer in self.prunable_layers]
        )

    def on_train_batch_begin(self, batch, logs=None):
        tf.keras.backend.batch_set_value(
            [(layer.global_step, self.step) for layer in self.prunable_layers]
        )

    def on_train_batch_end(self, batch, logs=None):
        for layer in self.prunable_layers:
            layer.mask_updater.conditional_update(training=True)
        self.step = self.step + 1


@KerasModifierYAML()
class ConstantKSModifier(ScheduledModifier, PruningScheduler):
    """
    Holds the sparsity level and shape for a given param constant while training.
    Useful for transfer learning use cases.

    | Sample yaml:
    |   !ConstantKSModifier
    |       params: __ALL__
    |       start_epoch: 0.0
    |       end_epoch: 10.0
    |       log_types: __ALL__

    :param params: List of str names or regex patterns of names for the parameter
        variables to apply the KS modifier to. Regex patterns must be specified
        with the prefix 're:'. Can also use the token __ALL__ to specify all
        prunable layers and weights
    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param log_types: The loggers to allow the learning rate to be logged to,
        default is __ALL__
    """

    def __init__(
        self,
        params: Union[str, List[str]],
        start_epoch: float = -1,
        end_epoch: float = -1,
        log_types: Union[str, List[str]] = ALL_TOKEN,
    ):
        super(ConstantKSModifier, self).__init__(
            log_types=log_types,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            end_comparator=None,
        )
        self._params = validate_str_iterable(
            params, "{} for params".format(self.__class__.__name__)
        )  # type: List[str]
        self._layer_names = [get_layer_name_from_param(p) for p in self._params]
        self._masked_layers = []

        self._sparsity_scheduler = None
        self._mask_creator = load_mask_creator("unstructured")

    @ModifierProp()
    def params(self) -> Union[str, List[str]]:
        """
        :return: List of str for the variable names or regex patterns of names
            to apply the KS modifier to. Regex patterns must be specified with
            the prefix 're:'.
        """
        return self._params

    @params.setter
    def params(self, value: Union[str, List[str]]):
        """
        :param value: List of str for the variable names or regex patterns of names
            to apply the KS modifier to. Regex patterns must be specified with
            the prefix 're:'.
        """
        self._params = value

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
    def sparsity(self) -> Union[None, tf.Tensor]:
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
            mask = tf.cast(tf.not_equal(tensor, 0.0), tensor.dtype)
            self._sparsity = tf.math.reduce_sum(1.0 - mask).numpy() / tf.size(tensor)
        elif is_end_step:
            mask = tf.ones_like(tensor)
            self._sparsity = 0.0
        else:
            self._sparsity = None
            mask = None
        return self._update_ready, self._sparsity, mask

    def _create_sparsity_scheduler(self, steps_per_epoch):
        begin_step, end_step = self.start_end_steps(steps_per_epoch, after_optim=False)
        sparsity_scheduler = SparsityFreezer(begin_step, end_step)
        return sparsity_scheduler

    def _clone_layer(self, layer: tf.keras.layers.Layer):
        cloned_layer = layer
        if layer.name in self.layer_names:  # TODO: handle regex params
            cloned_layer = MaskedLayer(
                layer, self._sparsity_scheduler, self._mask_creator, name=layer.name
            )
            self._masked_layers.append(cloned_layer)
        return cloned_layer

    def modify(
        self,
        model,
        optimizer,
        steps_per_epoch: int,
        input_tensors: tf.Tensor = None,
    ):
        """
        Modify model and optimizer

        :param model: a model to be modified
        :param optimizer: an optimizer to be modified
        :param steps_per_epoch: number of steps per epoch
        :param input_tensors: optional input tensors
        :return: modified model, optimizer and callbacks
        """
        model, optimizer, callback = super(ConstantKSModifier, self).modify(
            model, optimizer, steps_per_epoch, input_tensors=input_tensors
        )
        self._sparsity_scheduler = self._create_sparsity_scheduler(steps_per_epoch)
        cloned_model = tf.keras.models.clone_model(
            model,
            input_tensors,
            clone_function=self._clone_layer,
        )
        pruning_step_callback = KSModifierCallback(self._masked_layers)
        return cloned_model, optimizer, pruning_step_callback


@KerasModifierYAML()
class GradualKSModifier(ScheduledUpdateModifier):
    """
    Gradually applies kernel sparsity to a given variable or variables from
    init_sparsity until final_sparsity is reached over a given amount of time and
    applied with an interpolated function for each step taken.

    Applies based on magnitude pruning without any structure to the pruning.

    | Sample yaml:
    |   !GradualKSModifier
    |       params: __ALL__
    |       init_sparsity: 0.05
    |       final_sparsity: 0.8
    |       start_epoch: 0.0
    |       end_epoch: 10.0
    |       update_frequency: 1.0
    |       inter_func: cubic
    |       log_types: __ALL__
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
    :param log_types: The loggers to allow the learning rate to be logged to,
        default is __ALL__
    :param mask_type: String to define type of sparsity (options: ['unstructured',
        'channel', 'filter']), List to define block shape of a parameter's in and out
        channels, or a SparsityMaskCreator object. default is 'unstructured'
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
        log_types: Union[str, List[str]] = ALL_TOKEN,
        mask_type: Union[str, List[int], SparsityMaskCreator] = "unstructured",
        leave_enabled: bool = True,
    ):
        super(GradualKSModifier, self).__init__(
            log_types=log_types,
            start_epoch=start_epoch,
            min_start=-1.0,
            end_epoch=end_epoch,
            min_end=0.0,
            end_comparator=1,
            update_frequency=update_frequency,
            min_frequency=-1.0,
        )
        self._params = validate_str_iterable(
            params, "{} for params".format(self.__class__.__name__)
        )  # type: List[str]
        self._layer_names = [get_layer_name_from_param(p) for p in self._params]
        self._init_sparsity = init_sparsity
        self._final_sparsity = final_sparsity
        self._leave_enabled = convert_to_bool(leave_enabled)
        self._inter_func = inter_func
        self._mask_type = mask_type
        self._mask_creator = mask_type
        self._leave_enabled = convert_to_bool(leave_enabled)
        if not isinstance(mask_type, SparsityMaskCreator):
            self._mask_creator = load_mask_creator(mask_type)
        self._prune_op_vars = None
        self._update_ready = None
        self._sparsity = None
        self._mask_initializer = None

        self._masked_layers = []

        self.validate()

    @ModifierProp()
    def params(self) -> Union[str, List[str]]:
        """
        :return: List of str for the variable names or regex patterns of names
            to apply the KS modifier to. Regex patterns must be specified with
            the prefix 're:'.
        """
        return self._params

    @params.setter
    def params(self, value: Union[str, List[str]]):
        """
        :param value: List of str for the variable names or regex patterns of names
            to apply the KS modifier to. Regex patterns must be specified with
            the prefix 're:'.
        """
        self._params = value
        self.validate()

    @property
    def layer_names(self) -> List[str]:
        return self._layer_names

    @ModifierProp()
    def init_sparsity(self) -> float:
        """
        :return: The initial sparsity for the variable to start with at start_epoch
        """
        return self._init_sparsity

    @init_sparsity.setter
    def init_sparsity(self, value: float):
        """
        :param value: The initial sparsity for the variable to start with at start_epoch
        """
        self._init_sparsity = value
        self.validate()

    @ModifierProp()
    def final_sparsity(self) -> float:
        """
        :return: The final sparsity for the variable to end with at end_epoch
        """
        return self._final_sparsity

    @final_sparsity.setter
    def final_sparsity(self, value: float):
        """
        :param value: The final sparsity for the variable to end with at end_epoch
        """
        self._final_sparsity = value
        self.validate()

    @ModifierProp()
    def leave_enabled(self) -> bool:
        """
        :return: True to continue masking the weights after end_epoch,
            False to stop masking. Should be set to False if exporting
            the result immediately after or doing some other prune
        """
        return self._leave_enabled

    @leave_enabled.setter
    def leave_enabled(self, value: bool):
        """
        :param value: True to continue masking the weights after end_epoch,
            False to stop masking. Should be set to False if exporting the result
            immediately after or doing some other prune
        """
        self._leave_enabled = value
        self.validate()

    @ModifierProp()
    def inter_func(self) -> str:
        """
        :return: The type of interpolation function to use:
            [linear, cubic, inverse_cubic]
        """
        return self._inter_func

    @inter_func.setter
    def inter_func(self, value: str):
        """
        :param value: The type of interpolation function to use:
            [linear, cubic, inverse_cubic]
        """
        self._inter_func = value
        self.validate()

    @ModifierProp()
    def mask_type(self) -> Union[str, List[int], SparsityMaskCreator]:
        """
        :return: the SparsityMaskCreator object used
        """
        return self._mask_type

    @mask_type.setter
    def mask_type(self, value: Union[str, List[int], SparsityMaskCreator]):
        """
        :param value: the SparsityMaskCreator object to use
        """
        self._mask_type = value
        self._mask_creator = value
        if not isinstance(value, SparsityMaskCreator):
            self._mask_creator = load_mask_creator(value)

    @ModifierProp()
    def leave_enabled(self) -> bool:
        """
        :return: True to continue masking the weights after end_epoch,
            False to stop masking. Note, if set as False, sparsity will not be enforced
            and the model will likely deviate from the sparse solution
        """
        return self._leave_enabled

    @leave_enabled.setter
    def leave_enabled(self, value: bool):
        """
        :param value: True to continue masking the weights after end_epoch,
            False to stop masking. Note, if set as False, sparsity will not be enforced
            and the model will likely deviate from the sparse solution
        """
        self._leave_enabled = value

    @ModifierProp(serializable=False)
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

    @property
    def update_ready(self):
        """
        :return: the created update_ready tensor for setting the pruning ops
            if create_ops has been called, else None
        """
        return self._update_ready

    @property
    def sparsity(self) -> Union[None, tf.Tensor]:
        """
        :return: the created sparsity tensor for setting the pruning ops
            if create_ops has been called, else None
        """
        return self._sparsity

    def validate(self):
        """
        Validate the values of the params for the current instance are valid
        """

        if not self._leave_enabled:
            raise ValueError(
                "leave_enabled == True is only supported for {}".format(
                    self.__class__.__name__
                )
            )

        if not isinstance(self._init_sparsity, float):
            raise TypeError(
                "init_sparsity must be of float type for {}".format(
                    self.__class__.__name__
                )
            )

        if not 0.0 <= self._init_sparsity <= 1.0:
            raise ValueError(
                (
                    "init_sparsity value must be in the range"
                    " [0.0, 1.0], given {} for {}"
                ).format(self._init_sparsity, self.__class__.__name__)
            )

        if not isinstance(self._final_sparsity, float):
            raise TypeError(
                "final_sparsity must be of float type for {}".format(
                    self.__class__.__name__
                )
            )

        if not 0.0 <= self._final_sparsity <= 1.0:
            raise ValueError(
                (
                    "final_sparsity value must be in the range"
                    " [0.0, 1.0], given {} for {}"
                ).format(self._init_sparsity, self.__class__.__name__)
            )

        interpolation_funcs = ["linear", "cubic", "inverse_cubic"]

        if self._inter_func not in interpolation_funcs:
            raise ValueError(
                (
                    "{} is not a supported inter_func in layers_settings,"
                    " available are {} for {}"
                ).format(self._inter_func, interpolation_funcs, self.__class__.__name__)
            )

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

    def _clone_layer(self, layer: tf.keras.layers.Layer):
        cloned_layer = layer
        if (
            layer.name in self.layer_names
        ):  # TODO: handle regex params --- see create_ops in TF version
            cloned_layer = MaskedLayer(
                layer, self._sparsity_scheduler, self._mask_creator, name=layer.name
            )
            self._masked_layers.append(cloned_layer)
        return cloned_layer

    def modify(
        self, model, optimizer, steps_per_epoch: int, input_tensors: tf.Tensor = None
    ):
        """
        Modify model and optimizer, and provide callbacks to process the model

        :param model: a model to be modified with prunable layers wrapped by masks
        :param optimizer: an optimizer to be modified; in this case, no change to it
        :param steps_per_epoch: number of steps per epoch
        :param input_tensors: optional input tensors
        :return: modified model, optimizer and callbacks
        """

        # TODO: incorporate the returned callback into the final callbacks to return
        model, optimizer, callback = super(GradualKSModifier, self).modify(
            model, optimizer, steps_per_epoch, input_tensors=input_tensors
        )

        self._sparsity_scheduler = self._create_sparsity_scheduler(steps_per_epoch)

        # Clone model and additional set up
        cloned_model = tf.keras.models.clone_model(
            model,
            input_tensors,
            clone_function=self._clone_layer,
        )
        cloned_model.optimizer = optimizer

        # Pruning step call back and additional set up
        pruning_step_callback = KSModifierCallback(self._masked_layers)
        pruning_step_callback.set_model(cloned_model)

        return cloned_model, optimizer, pruning_step_callback
