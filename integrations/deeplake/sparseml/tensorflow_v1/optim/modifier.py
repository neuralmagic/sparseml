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
Contains base code related to modifiers: objects that modify some aspect
of the training process for a model.
For example, learning rate schedules or kernel sparsity (weight pruning)
are implemented as modifiers.
"""

from typing import Any, Dict, List, Tuple, Union

from sparseml.optim import (
    BaseModifier,
    BaseScheduled,
    BaseUpdate,
    ModifierProp,
    ModifierYAML,
)
from sparseml.tensorflow_v1.utils import tf_compat
from sparseml.utils import TENSORFLOW_V1_FRAMEWORK


__all__ = [
    "EXTRAS_KEY_LEARNING_RATE",
    "EXTRAS_KEY_SUMMARIES",
    "EXTRAS_KEY_VAR_LIST",
    "NM_RECAL",
    "ModifierProp",
    "TENSORFLOW_V1_FRAMEWORK",
    "TensorFlowModifierYAML",
    "Modifier",
    "ScheduledModifier",
    "ScheduledUpdateModifier",
    "ModifierSessionRunHook",
]


EXTRAS_KEY_LEARNING_RATE = "learning_rate"
EXTRAS_KEY_SUMMARIES = "summaries"
EXTRAS_KEY_VAR_LIST = "var_list"

NM_RECAL = "nm_recal"


class TensorFlowModifierYAML(ModifierYAML):
    """
    A decorator to handle making a TensorFlow modifier class YAML ready.
    IE it can be loaded in through the yaml plugin easily.
    """

    def __init__(self):
        super().__init__(TENSORFLOW_V1_FRAMEWORK)


class Modifier(BaseModifier):
    """
    Base modifier class that all TensorFlow modifiers should derive themselves from.
    Handles setting up the expected contracts for modifying graphs, ops, and extras.

    | Modifiers are expected to implement up to 3 different functions for TensorFlow:
    |  - create_ops - inject ops into the graph before the training begins
    |  - create_extras - create extras like learning rate controls before training
    |  - complete_graph - finalize the graph after training has completed
    |
    | Life cycle:
    |   - create model graph
    |   - manager.create_ops()
    |   - manager.create_extras()
    |   - train graph
    |   - manager.complete_graph()
    |   - export graph

    :param kwargs: standard key word args, used to support multi inheritance
    """

    @staticmethod
    def load_list(yaml_str: str):
        """
        :param yaml_str: a string representation of the yaml syntax to
            load modifiers from
        :return: the loaded modifiers list
        """
        return Modifier.load_framework_list(yaml_str, TENSORFLOW_V1_FRAMEWORK)

    @staticmethod
    def load_obj(yaml_str: str):
        """
        :param yaml_str:  a string representation of the yaml syntax to
            load a modifier from
        :return: the loaded modifier object
        """
        return Modifier.load_framework_obj(yaml_str, TENSORFLOW_V1_FRAMEWORK)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_group(self) -> Any:
        """
        Function to be override by a subclass indicating the modifier container
        into which the subclass should be combined
        As an example, the two learning rate modifier classes SetLearningRateModifier
        and LearningRateModifier return GroupLearningRateModifier, meaning that
        a sequence of those LR modifier instances are grouped into the
        GroupLearningRateModifier, which is where the final learning rate is computed
        """
        return None

    def modify_estimator(
        self, estimator: tf_compat.estimator.Estimator, steps_per_epoch: int
    ):
        """
        Modify a tf Estimator. Overrides the model_fn so that on invocation
        it creates the original graph and then calls into create_ops.
        Additionally will recreate the Scaffold with a new Save instance
        to save all variables in the modified graph.

        Note, learning_rate and other specific tensors that needed to be
        retrieved from the extras in create_ops and passed to another implementation
        will not work with this flow.

        :param estimator: the tf Estimator to modify
        :param steps_per_epoch: number of steps per training epoch
        """
        orig_model_func = (
            estimator._model_fn
        )  # type: Callable[[Any...], tf_compat.estimator.EstimatorSpec]  # noqa

        def _model_func(
            features: Dict[str, tf_compat.Tensor],
            labels: Dict[str, tf_compat.Tensor],
            mode: tf_compat.estimator.ModeKeys,
            params: Dict[str, Any],
        ):
            spec = orig_model_func(
                features=features, labels=labels, mode=mode, params=params
            )
            graph = tf_compat.get_default_graph()

            with graph.as_default():
                global_step = tf_compat.train.get_or_create_global_step()

            mod_ops, mod_extras = self.create_ops(steps_per_epoch, global_step, graph)
            hook = ModifierSessionRunHook(self, steps_per_epoch, mod_ops, mod_extras)
            replace_kwargs = {}

            if mode == tf_compat.estimator.ModeKeys.TRAIN:
                replace_kwargs = {"training_hooks": [hook]}

                if spec.training_hooks:
                    replace_kwargs["training_hooks"].extend(spec.training_hooks)

            orig_saver = spec.scaffold.saver
            saver = tf_compat.train.Saver(
                var_list=None,
                reshape=orig_saver._reshape,
                sharded=orig_saver._sharded,
                max_to_keep=orig_saver._max_to_keep,
                keep_checkpoint_every_n_hours=orig_saver._keep_checkpoint_every_n_hours,
                name=orig_saver._name,
                restore_sequentially=orig_saver._restore_sequentially,
                pad_step_number=orig_saver._pad_step_number,
                save_relative_paths=orig_saver._save_relative_paths,
                filename=orig_saver._filename,
            )
            replace_kwargs["scaffold"] = tf_compat.train.Scaffold(
                saver=saver, copy_from_scaffold=spec.scaffold
            )
            spec = spec._replace(**replace_kwargs)

            return spec

        estimator._model_fn = _model_func

    def create_ops(
        self,
        steps_per_epoch: int,
        global_step: tf_compat.Tensor,
        graph: tf_compat.Graph,
    ) -> Tuple[List[Union[tf_compat.Tensor, tf_compat.Operation]], Dict[str, Any]]:
        """
        Create modifying operations and tensors in the graph.

        | Returns a tuple containing:
        |   - modifying ops that should be run in a session on each global step.
        |   - named extras (ops / tensors) created in the graph that can be used
        |     by other ops such as a learning rate for the optimizer

        :param steps_per_epoch: the number of steps (batches) per training epoch
        :param global_step: the global step used while training
        :param graph: the graph to be modified
        :return: a tuple (list of ops, dict of named ops / tensors)
            to be run or used for modifying the training process
        """
        self._initialized = True

        return [], {}

    def initialize_session(self, sess: tf_compat.Session):
        """
        Initialize any state for a session such as variables.

        :param sess: the session to use for initializing
        """
        if not self._initialized:
            raise RuntimeError(
                "create_ops for modifier must be called before initialize_session"
            )

    def complete_graph(self, graph: tf_compat.Graph, sess: tf_compat.Session):
        """
        Complete modifying the graph. Should be called after modifying is complete.
        Cleans up any ops that should be removed or reordered.

        :param graph: the modified graph that should be completed and cleaned
        :param sess: the session to use for completing the modified graph
        :return: the cleaned graph
        """
        if not self._initialized:
            raise RuntimeError(
                "create_ops for modifier must be called before complete_graph"
            )


class ScheduledModifier(Modifier, BaseScheduled):
    """
    The base scheduled update modifier implementation, all scheduled modifiers should
    inherit from this class.
    Offers convenient properties needed for scheduled update modifiers:
    start_epoch, end_epoch


    | Modifiers are expected to implement up to 3 different functions for TensorFlow:
    |  - create_ops - inject ops into the graph before the training begins
    |  - create_extras - create extras like learning rate controls before training
    |  - complete_graph - finalize the graph after training has completed
    |
    | Life cycle:
    |   - create model graph
    |   - manager.create_ops()
    |   - manager.create_extras()
    |   - train graph
    |   - manager.complete_graph()
    |   - export graph

    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param min_start: The minimum acceptable value for start_epoch, default -1
    :param min_end: The minimum acceptable value for end_epoch, default 0
    :param end_comparator: integer value representing how the end_epoch should be
        compared to start_epoch.
        if == None, then end_epoch can only be set to what its initial value was.
        if == -1, then end_epoch can be -1, equal, or greater than start_epoch.
        if == 0, then end_epoch can be equal to or greater than start_epoch.
        if == 1, then end_epoch can only be greater than start_epoch.
    :param kwargs: standard key word args, used to support multi inheritance
    """

    def __init__(
        self,
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
        min_start: float = -1.0,
        min_end: float = -1.0,
        end_comparator: Union[int, None] = 0,
        **kwargs,
    ):
        super().__init__(
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            min_start=min_start,
            min_end=min_end,
            end_comparator=end_comparator,
            **kwargs,
        )

    def start_end_steps(
        self, steps_per_epoch: int, after_optim: bool
    ) -> Tuple[int, int]:
        """
        Calculate the start and end steps for this modifier given a certain
        amount of steps per epoch

        :param steps_per_epoch: the number of steps (or batches) taken per epoch
        :param after_optim: True if the start and end are for an operation after
            the optimizer update step has run, False for before
        :return: a tuple containing (the converted start step,
            the converted end step)
        """
        start_step = (
            round(self._start_epoch * steps_per_epoch) if self.start_epoch >= 0.0 else 0
        )
        end_step = (
            round(self._end_epoch * steps_per_epoch) - 1
            if self.end_epoch >= 0.0
            else -1
        )

        if after_optim:
            start_step += 1

            if end_step > -1:
                end_step += 1

        return start_step, end_step


class ScheduledUpdateModifier(ScheduledModifier, BaseUpdate):
    """
    The base scheduled update modifier implementation,
    all scheduled update modifiers should inherit from this class.
    Offers convenient properties needed for scheduled update modifiers: update_frequency


    | Modifiers are expected to implement up to 3 different functions for TensorFlow:
    |  - create_ops - inject ops into the graph before the training begins
    |  - create_extras - create extras like learning rate controls before training
    |  - complete_graph - finalize the graph after training has completed
    |
    | Life cycle:
    |   - create model graph
    |   - manager.create_ops()
    |   - manager.create_extras()
    |   - train graph
    |   - manager.complete_graph()
    |   - export graph

    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param min_start: The minimum acceptable value for start_epoch, default -1
    :param min_end: The minimum acceptable value for end_epoch, default 0
    :param end_comparator: integer value representing how the end_epoch should be
        compared to start_epoch.
        if == -1, then end_epoch can be -1, equal, or greater than start_epoch.
        if == 0, then end_epoch can be equal to or greater than start_epoch.
        if == 1, then end_epoch can only be greater than start_epoch.
    :param update_frequency: The number of epochs or fraction of epochs to
        update at between start and end
    :param min_frequency: The minimum acceptable value for update_frequency, default -1
    :param kwargs: standard key word args, used to support multi inheritance
    """

    def __init__(
        self,
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
        min_start: float = -1.0,
        min_end: float = -1.0,
        end_comparator: int = 0,
        update_frequency: float = -1.0,
        min_frequency: float = -1.0,
        **kwargs,
    ):
        super().__init__(
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            min_start=min_start,
            min_end=min_end,
            end_comparator=end_comparator,
            update_frequency=update_frequency,
            min_frequency=min_frequency,
            **kwargs,
        )

    def update_frequency_steps(self, steps_per_epoch: int) -> int:
        """
        Calculate the update frequency steps for this modifier given a certain
        amount of steps per epoch

        :param steps_per_epoch: the number of steps (or batches) taken per epoch
        :return: a tuple containing (the converted start step,
            the converted end step)
        """
        update_frequency_steps = round(self._update_frequency * steps_per_epoch)

        return update_frequency_steps


def epoch_to_steps(epoch: float, steps_per_epoch: int, min_epoch: float = 0.0) -> int:
    """
    :param epoch: the (fractional) epoch to convert to the proper number of steps
    :param steps_per_epoch: number of steps (batches) taken per epoch while training
    :param min_epoch: if the epoch is less than this, will be set to it. Default 0
    :return: the number of steps representing the epoch and state of the epoch
    """

    if epoch < min_epoch:
        epoch = min_epoch

    return round(steps_per_epoch * epoch)


class ModifierSessionRunHook(tf_compat.train.SessionRunHook):
    """
    A session run hook for the tf Estimator flow.
    Used to integrate so that any extra ops for modifying the graph
    can be executed each on each step of the estimator training process.

    :param modifier: the modifier to run the hook for
    :param steps_per_epoch: number of steps (or batches) taken per epoch
    :param mod_ops: the ops returned from calling create_ops on the modifier
    :param mod_extras: the extras returned from calling create_ops on the modifier
    """

    def __init__(
        self,
        mod_ops: List[Union[tf_compat.Tensor, tf_compat.Operation]],
    ):
        self._mod_ops = mod_ops

    def after_run(self, run_context, run_values):
        """
        Called by the estimator after each call to run()

        :param run_context: run_context passed in during training
        :param run_values: a SessionRunValues object passed in during training
        """
        if self._mod_ops:
            sess = run_context.session
            sess.run(self._mod_ops)
