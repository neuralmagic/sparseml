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
Learning rate modifiers for TensorFlow models
"""
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

from sparseml.sparsification import LearningRateModifier as BaseLearningRateModifier
from sparseml.sparsification import (
    SetLearningRateModifier as BaseSetLearningRateModifier,
)
from sparseml.tensorflow_v1.optim.modifier import (
    EXTRAS_KEY_LEARNING_RATE,
    EXTRAS_KEY_SUMMARIES,
    NM_RECAL,
    ScheduledModifier,
    ScheduledUpdateModifier,
    TensorFlowModifierYAML,
)
from sparseml.tensorflow_v1.optim.schedule_lr import (
    multi_step_lr_schedule,
    step_lr_schedule,
)
from sparseml.tensorflow_v1.utils import tf_compat


__all__ = [
    "SetLearningRateModifier",
    "LearningRateModifier",
    "GroupLearningRateModifier",
]


def _add_lr_extras(
    mod_extras: Dict,
    learning_rate: tf_compat.Tensor,
):
    mod_extras[EXTRAS_KEY_LEARNING_RATE] = learning_rate
    mod_extras[EXTRAS_KEY_SUMMARIES] = [
        tf_compat.summary.scalar("Train/learning_rate", learning_rate)
    ]


class GroupLearningRateModifier(ScheduledModifier):
    """
    Combining multiple LR modifiers, correctly compute the learning rate

    :param lr_modifiers: List of LR modifiers to combine
    """

    def __init__(self, lr_modifiers: List[ScheduledModifier]):
        assert len(lr_modifiers) > 0
        lr_modifiers = deepcopy(lr_modifiers)
        lr_modifiers = sorted(lr_modifiers, key=lambda m: m.start_epoch, reverse=True)
        start_epoch = min([mod.start_epoch for mod in lr_modifiers])
        end_epoch = max([mod.end_epoch for mod in lr_modifiers])

        super().__init__(
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            end_comparator=-1,
        )

        self._lr_modifiers = lr_modifiers

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
        name_scope = "{}/{}".format(NM_RECAL, self.__class__.__name__)

        with graph.as_default():
            with tf_compat.name_scope(name_scope):
                pred_fn_pairs = []
                global_step = tf_compat.cast(global_step, tf_compat.int64)

                for index, child in enumerate(self._lr_modifiers):
                    with tf_compat.name_scope(str(index)):
                        _, child_extras = child.create_ops(
                            steps_per_epoch, global_step, graph
                        )
                        child_lr = child_extras[EXTRAS_KEY_LEARNING_RATE]
                        child_start_step, _ = child.start_end_steps(
                            steps_per_epoch, after_optim=False
                        )
                        child_select = tf_compat.greater_equal(
                            global_step,
                            tf_compat.constant(child_start_step, tf_compat.int64),
                            name="active",
                        )
                        pred_fn_pairs.append((child_select, lambda lr=child_lr: lr))

                learning_rate = tf_compat.case(pred_fn_pairs)
                _add_lr_extras(mod_extras, learning_rate)

        return mod_ops, mod_extras


@TensorFlowModifierYAML()
class SetLearningRateModifier(BaseSetLearningRateModifier, ScheduledModifier):
    """
    Modifier to set the learning rate to a specific value at a certain point
    in the training process. Once that point is reached, will update the optimizer's
    params with the learning rate

    | Sample yaml:
    |    !SetLearningRateModifier
    |        start_epoch: 0.0
    |        learning_rate: 0.001

    :param learning_rate: The learning rate to use once this modifier starts
    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: unused and should not be set
    """

    def __init__(
        self,
        learning_rate: float,
        start_epoch: float = -1,
        end_epoch: float = -1,
    ):
        super(SetLearningRateModifier, self).__init__(
            learning_rate=learning_rate,
            start_epoch=start_epoch,
            end_epoch=-1,
            end_comparator=None,
        )

    def get_group(self):
        """
        :return: The group modifier class into which this modifier needs to be combined
        """
        return GroupLearningRateModifier

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
        :return: a tuple (empty list of ops,
            dict of learning rate and logging summaries)
        """
        mod_ops, mod_extras = super().create_ops(graph, steps_per_epoch, global_step)
        name_scope = "{}/{}".format(NM_RECAL, self.__class__.__name__)

        with graph.as_default():
            with tf_compat.name_scope(name_scope):
                learning_rate = tf_compat.constant(
                    self.learning_rate, tf_compat.float32, name="learning_rate"
                )

            _add_lr_extras(mod_extras, learning_rate)

        return mod_ops, mod_extras


@TensorFlowModifierYAML()
class LearningRateModifier(BaseLearningRateModifier, ScheduledUpdateModifier):
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

    :param lr_class: The name of the lr scheduler class to use:
        [StepLR, MultiStepLR, ExponentialLR]
    :param lr_kwargs: The dictionary of keyword arguments to pass to the constructor
        for the lr_class
    :param init_lr: The initial learning rate to use once this modifier starts
    :param start_epoch: The epoch to start the modifier at
        (set to -1.0 so it starts immediately)
    :param end_epoch: The epoch to end the modifier at,
        (set to -1.0 so it doesn't end)
    :param update_frequency: unused and should not be set
    """

    def __init__(
        self,
        lr_class: str,
        lr_kwargs: Dict,
        init_lr: float,
        start_epoch: float,
        end_epoch: float = -1.0,
        update_frequency: float = -1.0,
    ):
        super(LearningRateModifier, self).__init__(
            lr_class=lr_class,
            lr_kwargs=lr_kwargs,
            init_lr=init_lr,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=-1,
            end_comparator=-1,
        )

    def get_group(self):
        """
        :return: The group modifier class into which this modifier needs to be combined
        """
        return GroupLearningRateModifier

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
        name_scope = "{}/{}".format(NM_RECAL, self.__class__.__name__)

        with graph.as_default():
            with tf_compat.name_scope(name_scope):
                lr_class, lr_kwargs = self.corrected_lr_info(
                    steps_per_epoch, self.start_epoch, self.end_epoch
                )
                start_step, end_step = self.start_end_steps(
                    steps_per_epoch, after_optim=False
                )

                if lr_class == "StepLR":
                    learning_rate = step_lr_schedule(
                        global_step,
                        start_step,
                        end_step,
                        lr_kwargs["step_size"],
                        self.init_lr,
                        lr_kwargs["gamma"],
                    )
                elif lr_class == "MultiStepLR":
                    learning_rate = multi_step_lr_schedule(
                        global_step,
                        start_step,
                        lr_kwargs["milestones"],
                        self.init_lr,
                        lr_kwargs["gamma"],
                    )
                else:
                    raise ValueError(
                        "unrecognized lr_class given of {}".format(lr_class)
                    )

            _add_lr_extras(mod_extras, learning_rate)

        return mod_ops, mod_extras
