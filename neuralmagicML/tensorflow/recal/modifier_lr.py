"""
Learning rate modifiers for TensorFlow models
"""
from typing import Any, Dict, Union, List, Tuple, Optional

from neuralmagicML.tensorflow.utils import tf_compat
from tensorflow.keras.optimizers.schedules import (
    ExponentialDecay,
    PiecewiseConstantDecay,
)

from neuralmagicML.utils import ALL_TOKEN, convert_to_bool
from neuralmagicML.tensorflow.recal.modifier import (
    EXTRAS_KEY_LEARNING_RATE,
    EXTRAS_KEY_SUMMARIES,
    NM_RECAL,
    ModifierProp,
    ScheduledModifier,
    ScheduledUpdateModifier,
    TensorFlowModifierYAML,
)

__all__ = [
    "SetLearningRateModifier",
    "LearningRateModifier",
    "GroupLearningRateModifier",
]


class GroupLearningRateModifier(ScheduledModifier):
    """
    Combining multiple LR modifiers, correctly compute the learning rate

    :param lr_modifiers: List of LR modifiers to combine
    :param log_types: The loggers to allow the learning rate to be logged to,
        default is __ALL__
    """

    def __init__(self, lr_modifiers: List[ScheduledModifier]):
        super().__init__()
        self._lr_modifiers = sorted(lr_modifiers, key=lambda mod: mod.start_epoch)
        if self._lr_modifiers[0].start_epoch != 0:
            raise ValueError("The first start epoch must be zero")

        # Log types of this container modifier is promoted from those
        # of the individials
        self._log_types = None
        for mod in self._lr_modifiers:
            if mod.log_types == ALL_TOKEN or "tensorboard" in mod.log_types:
                self._log_types = mod.log_types
                break
        for mod in self._lr_modifiers:
            mod.log_types = None

    def create_ops(
        self,
        steps_per_epoch: int,
        global_step: tf_compat.Variable,
        graph: tf_compat.Graph,
    ) -> Tuple[List[Union[tf_compat.Tensor, tf_compat.Operation]], Dict[str, Any]]:
        """
        Create switch case computing the learning rate at a given global step and
        extras created by individual LR modifiers

        :param steps_per_epoch: the number of steps per training epoch
        :param global_step: the global step used while training
        :param graph: the graph to be modified
        :return: a tuple (list of empty ops, dict of named ops/tensors for learning
            rate and summaries as extras)
        """
        mod_ops, mod_extras = super().create_ops(graph, steps_per_epoch, global_step)
        lr_switch_case = None
        name_scope = "{}/{}".format(NM_RECAL, self.__class__.__name__)
        with graph.as_default():
            ops_and_extras = [
                mod.create_ops(steps_per_epoch, global_step, graph)
                for mod in self._lr_modifiers
            ]  # List[Tuple[List, Dict]]

            with tf_compat.name_scope(name_scope):
                extras = [
                    _extras for (_, _extras) in ops_and_extras
                ]  # List[Dict[str, Any]]
                pred_fn_pairs = []
                for i in range(len(extras)):
                    current_start_step = tf_compat.constant(
                        round(self._lr_modifiers[i].start_epoch * steps_per_epoch),
                        dtype=tf_compat.int64,
                    )
                    if i == len(extras) - 1:
                        pred = tf_compat.greater_equal(global_step, current_start_step)
                    else:
                        next_start_step = tf_compat.constant(
                            round(
                                self._lr_modifiers[i + 1].start_epoch * steps_per_epoch
                            ),
                            dtype=tf_compat.int64,
                        )
                        pred = tf_compat.logical_and(
                            tf_compat.greater_equal(global_step, current_start_step),
                            tf_compat.less(global_step, next_start_step),
                        )

                    def make_fn(i):
                        return lambda: extras[i][EXTRAS_KEY_LEARNING_RATE]

                    pred_fn_pairs.append((pred, make_fn(i)))
                lr_switch_case = tf_compat.case(pred_fn_pairs)

            if self.log_types == ALL_TOKEN or "tensorboard" in self.log_types:
                mod_extras[EXTRAS_KEY_SUMMARIES] = [
                    tf_compat.summary.scalar("learning_rate", lr_switch_case)
                ]

            mod_extras[EXTRAS_KEY_LEARNING_RATE] = lr_switch_case

        return (mod_ops, mod_extras)


@TensorFlowModifierYAML()
class SetLearningRateModifier(ScheduledModifier):
    """
    Modifier to set the learning rate to a specific value at a certain point
    in the training process. Once that point is reached, will update the optimizer's
    params with the learning rate

    | Sample yaml:
    |    !SetLearningRateModifier
    |        start_epoch: 0.0
    |        learning_rate: 0.001
    |        log_types: __ALL__

    :param learning_rate: The learning rate to use once this modifier starts
    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: unused and should not be set
    :param log_types: The loggers to allow the learning rate to be logged to,
        default is __ALL__
    :param constant_logging: True to constantly log on every step, False to only
        log on an LR change, default True
    """

    def __init__(
        self,
        learning_rate: float,
        start_epoch: float = -1,
        end_epoch: float = -1,
        log_types: Union[str, List[str]] = ALL_TOKEN,
        constant_logging: bool = True,
    ):
        super().__init__(
            log_types=log_types,
            start_epoch=start_epoch,
            end_epoch=-1,
            end_comparator=None,  # end_epoch must stay at -1
        )

        self._start_epoch = start_epoch
        self._end_epoch = end_epoch
        self._update_ready = None
        self._learning_rate_val = learning_rate
        self._learning_rate_var = None
        self._constant_logging = convert_to_bool(constant_logging)
        self._last_logged_lr = None

    def get_group(self):
        """
        :return: The group modifier class into which this modifier needs to be combined
        """
        return GroupLearningRateModifier

    @ModifierProp()
    def start_epoch(self) -> float:
        """
        :return: The start epoch
        """
        return self._start_epoch

    @start_epoch.setter
    def start_epoch(self, value: float):
        """
        :param value: The new value for start_epoch
        """
        self._start_epoch = value
        self.validate_schedule()

    @ModifierProp()
    def log_types(self) -> Union[str, List[str]]:
        """
        :return: The logging types
        """
        return self._log_types

    @log_types.setter
    def log_types(self, value: Union[str, List[str]]):
        """
        :param value: The logging types to use
        """
        self._log_types = value

    @ModifierProp()
    def learning_rate(self) -> float:
        """
        :return: The learning rate to use once this modifier starts
        """
        return self._learning_rate_val

    def validate_schedule(self):
        """
        Validate the schedule values of the params for the current instance are valid
        """
        super().validate_schedule()
        if self._end_epoch != -1.0:
            raise ValueError(
                "end_epoch of {} must be equal to -1.0 for {}".format(
                    self._end_epoch, self.__class__.__name__
                )
            )

    def create_ops(
        self,
        steps_per_epoch: int,
        global_step: Optional[tf_compat.Variable],
        graph: Optional[tf_compat.Graph],
    ) -> Tuple[List[Union[tf_compat.Tensor, tf_compat.Operation]], Dict[str, Any]]:
        """
        Create ops to set the learning rate at a given value if the global step reaches
        a given value

        :param steps_per_epoch: the number of steps (batches) per training epoch
        :param global_step: the global step used while training
        :param graph: the graph to be modified
        :return: a tuple (empty list of ops, dict of learning rate and logging summaries)
        """
        mod_ops, mod_extras = super().create_ops(graph, steps_per_epoch, global_step)

        name_scope = "{}/{}".format(NM_RECAL, self.__class__.__name__)
        with graph.as_default():
            with tf_compat.name_scope(name_scope):
                self._learning_rate_const = tf_compat.constant(
                    self._learning_rate_val,
                    dtype=tf_compat.float32,
                    name=EXTRAS_KEY_LEARNING_RATE,
                )
                begin_step = round(self._start_epoch * steps_per_epoch)
                lr_assign_op = tf_compat.case(
                    [
                        (
                            tf_compat.greater_equal(global_step, begin_step),
                            lambda: self._learning_rate_const,
                        )
                    ]
                )

            mod_extras = {}

            if self.log_types is not None:
                if self._log_types == ALL_TOKEN or "tensorboard" in self._log_types:
                    mod_extras[EXTRAS_KEY_SUMMARIES] = tf_compat.summary.scalar(
                        "SetLR Modifier/learning_rate", lr_assign_op
                    )

            mod_extras[EXTRAS_KEY_LEARNING_RATE] = lr_assign_op

        return (mod_ops, mod_extras)


@TensorFlowModifierYAML()
class LearningRateModifier(ScheduledUpdateModifier):
    """
    Modifier to set the learning rate to follow specific schedulers
    within a period of epochs.
    The following schedulers are current supported: ExponentialLR,
    StepLR, MultiStepLR

    | Sample yaml:
    |    !LearningRateModifier
    |        lr_class: ExponentialDecay
    |        lr_kwargs:
    |            initial_learning_rate: 0.01
    |            decay_steps: 10000
    |            decay_rate: 0.96
    |        start_epoch: 0.0
    |        end_epoch: 10.0
    |        log_types: __ALL__

    :param lr_class: The name of the lr scheduler class to use
    :param lr_kwargs: The dictionary of keyword arguments to pass to the
        constructor for the lr_class
    :param init_lr: The initial learning rate to use once this modifier starts
    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param update_frequency: unused
    :param log_types: The loggers to allow the learning rate to be logged to,
        default is __ALL__
    """

    supported_schedulers = ["ExponentialLR", "StepLR", "MultiStepLR"]

    def __init__(
        self,
        lr_class: str,
        lr_kwargs: Dict,
        init_lr: float,
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
        update_frequency: float = -1.0,
        log_types: Union[str, List[str]] = ALL_TOKEN,
    ):
        if lr_class not in LearningRateModifier.supported_schedulers:
            raise ValueError("Invalid learning rate schedules.")

        super().__init__(
            log_types=log_types,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=-1,
            end_comparator=1,  # end_epoch must be greater than start_epoch
        )

        self._init_lr = init_lr
        self._lr_class = lr_class
        self._lr_kwargs = lr_kwargs
        self._lr_scheduler = None
        self._start_epoch = start_epoch
        self._end_epoch = end_epoch if end_epoch >= 0 else None

    def _create_scheduler(self, steps_per_epoch):
        """
        Mapping from the supported schedulers into those built-in
        in Tensorflow

        :param steps_per_epoch: Number of steps per epoch

        :return: a built-in Tensorflow scheduler
        """
        if self._lr_class == "ExponentialLR":
            initial_learning_rate = self._init_lr
            decay_steps = 1  # Decayed to full decay rate in every single step
            decay_rate = self._lr_kwargs["gamma"]
            staircase = False
            return ExponentialDecay(
                initial_learning_rate, decay_steps, decay_rate, staircase=staircase
            )

        elif self._lr_class == "StepLR":
            initial_learning_rate = self._init_lr
            decay_steps = self._lr_kwargs["step"]
            decay_rate = self._lr_kwargs["gamma"]

            # After each full decay, the updated learning rate stays the same for a
            # number of steps defined by the decay_steps param
            staircase = True

            return ExponentialDecay(
                initial_learning_rate, decay_steps, decay_rate, staircase=staircase
            )

        elif self._lr_class == "MultiStepLR":
            gamma = self._lr_kwargs["gamma"]
            boundaries = [
                round(v * steps_per_epoch) for v in self._lr_kwargs["milestones"]
            ]
            values = [self._init_lr * (gamma ** i) for i in range(len(boundaries) + 1)]
            return PiecewiseConstantDecay(boundaries, values)

        else:
            raise ValueError("Invalid learning rate scheduler")

    def get_group(self):
        """
        :return: The group modifier class into which this modifier needs to be combined
        """
        return GroupLearningRateModifier

    @ModifierProp()
    def lr_class(self) -> str:
        """
        :return: The name of the lr scheduler class to use
        """
        return self._lr_class

    @ModifierProp()
    def lr_kwargs(self) -> Dict:
        """
        :return: The dictionary of keyword arguments to pass to the constructor
            for the lr_class
        """
        return self._lr_kwargs

    @ModifierProp()
    def start_epoch(self) -> float:
        """
        :return: The start epoch
        """
        return self._start_epoch

    @start_epoch.setter
    def start_epoch(self, value: float):
        """
        :param value: The new value for start_epoch
        """
        self._start_epoch = value
        self.validate_schedule()

    @ModifierProp()
    def end_epoch(self) -> float:
        """
        :return: The end epoch
        """
        return self._end_epoch

    @end_epoch.setter
    def end_epoch(self, value: float):
        """
        :param value: The new value for end_epoch
        """
        self._end_epoch = value
        self.validate_schedule()

    @ModifierProp()
    def log_types(self) -> Union[str, List[str]]:
        """
        :return: The log types
        """
        return self._log_types

    @log_types.setter
    def log_types(self, value: Union[str, List[str]]):
        """
        :param value: The logging types to use
        """
        self._log_types = value

    @ModifierProp()
    def init_lr(self) -> float:
        """
        :return: The initial learning rate
        """
        return self._init_lr

    def create_ops(
        self,
        steps_per_epoch: int,
        global_step: Optional[tf_compat.Variable],
        graph: Optional[tf_compat.Graph],
    ) -> Tuple[List[Union[tf_compat.Tensor, tf_compat.Operation]], Dict[str, Any]]:
        """
        Create ops to update the learning rate at the current global step

        :param steps_per_epoch: the number of steps (batches) per training epoch
        :param global_step: the global step used while training
        :param graph: the graph to be modified
        :return: a tuple (empty list of ops, dict of learning rate and summaries)
        """
        mod_ops, mod_extras = super().create_ops(graph, steps_per_epoch, global_step)

        if self._lr_scheduler is None:
            self._lr_scheduler = self._create_scheduler(steps_per_epoch)

        name_scope = "{}/{}".format(NM_RECAL, self.__class__.__name__)
        with graph.as_default():
            with tf_compat.name_scope(name_scope):
                begin_step = round(self._start_epoch * steps_per_epoch)
                begin_step_const = tf_compat.constant(begin_step, dtype=tf_compat.int64)
                end_step = round(self._end_epoch * steps_per_epoch)
                end_step_const = tf_compat.constant(end_step, dtype=tf_compat.int64)
                delta_steps = tf_compat.constant(
                    end_step - begin_step, dtype=tf_compat.int64
                )

                lr_switch_case = tf_compat.case(
                    [
                        (
                            # Condition: begin_step <= global_step < end_step
                            tf_compat.logical_and(
                                tf_compat.greater_equal(global_step, begin_step_const),
                                tf_compat.less(global_step, end_step_const),
                            ),
                            # LR: scheduler(global_step - begin_step)
                            lambda: self._lr_scheduler(
                                tf_compat.math.subtract(global_step, begin_step_const)
                            ),
                        )
                    ],
                    default=lambda: self._lr_scheduler(delta_steps),
                )

            mod_extras = {}

            if self.log_types is not None:
                if self._log_types == ALL_TOKEN or "tensorboard" in self._log_types:
                    mod_extras[EXTRAS_KEY_SUMMARIES] = tf_compat.summary.scalar(
                        "LR Modifier/learning_rate", lr_switch_case
                    )

            mod_extras[EXTRAS_KEY_LEARNING_RATE] = lr_switch_case

        return (mod_ops, mod_extras)
