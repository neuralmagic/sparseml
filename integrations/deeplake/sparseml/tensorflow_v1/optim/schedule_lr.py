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
Learning rate schedules implementations for TensorFlow
"""

from typing import List

from sparseml.tensorflow_v1.utils import tf_compat


__all__ = [
    "step_lr_schedule",
    "multi_step_lr_schedule",
]


def step_lr_schedule(
    global_step: tf_compat.Tensor,
    start_step: int,
    end_step: int,
    step_size: int,
    init_lr: float,
    gamma: float,
    name: str = "exponential_lr_schedule",
) -> tf_compat.Tensor:
    """
    Create an exponential learning rate schedule in the current graph.
    Multiplies init_lr by gamma after each step_size interval has passed.
    Ex: lr = init_lr * (gamma ** NUM_UPDATES)

    :param global_step: the global step used for training
    :param start_step: the step to start the exponential schedule on
    :param end_step: the step to end the exponential schedule on,
        can be set to -1 and in that event will continually update the LR
    :param step_size: the number of steps between each gamma update to the init_lr
    :param init_lr: the learning rate to start the schedule with
    :param gamma: the decay weight to decrease init_lr by after every step_size interval
    :param name: the name scope to create the graph under
    :return: the calculated learning rate tensor
    """
    with tf_compat.name_scope(name):
        global_step = tf_compat.cast(global_step, tf_compat.int64)
        max_updates = tf_compat.constant(
            (end_step - start_step) // step_size if end_step > 0 else -1,
            dtype=tf_compat.int64,
            name="max_updates",
        )
        start_step = tf_compat.constant(
            start_step, dtype=tf_compat.int64, name="start_step"
        )
        end_step = tf_compat.constant(end_step, dtype=tf_compat.int64, name="end_step")
        init_lr = tf_compat.constant(init_lr, dtype=tf_compat.float32, name="init_lr")
        step_size = tf_compat.constant(
            step_size, dtype=tf_compat.int64, name="step_size"
        )
        gamma = tf_compat.constant(gamma, dtype=tf_compat.float32, name="gamma")
        before = tf_compat.less(global_step, start_step, name="before")
        after = tf_compat.logical_and(
            tf_compat.greater_equal(global_step, end_step, name="after"),
            tf_compat.not_equal(end_step, tf_compat.constant(-1, tf_compat.int64)),
        )

        def _calc_lr():
            steps = tf_compat.subtract(global_step, start_step)
            updates = tf_compat.cond(
                after,
                lambda: max_updates,
                lambda: tf_compat.cast(
                    tf_compat.floor(tf_compat.divide(steps, step_size)),
                    tf_compat.int64,
                ),
            )
            mult_g = tf_compat.pow(gamma, tf_compat.cast(updates, tf_compat.float32))

            return tf_compat.multiply(init_lr, mult_g)

        learning_rate = tf_compat.cond(
            before, lambda: init_lr, _calc_lr, name="learning_rate"
        )

    return learning_rate


def multi_step_lr_schedule(
    global_step: tf_compat.Tensor,
    start_step: int,
    milestone_steps: List[int],
    init_lr: float,
    gamma: float,
    name: str = "multi_step_lr_schedule",
):
    """
    Create a multi step learning rate schedule in the current graph.
    Multiplies init_lr by gamma after each milestone has passed.
    Ex: lr = init_lr * (gamma ** NUM_UPDATES)

    :param global_step: the global step used for training
    :param start_step: the step to start the exponential schedule on
    :param milestone_steps: a list of steps to decrease the learning rate at,
        these are the number of steps that must pass after start_step to decrease lr
    :param init_lr: the learning rate to start the schedule with
    :param gamma: the decay weight to decrease init_lr by after every step_size interval
    :param name: the name scope to create the graph under
    :return: the calculated learning rate tensor
    """
    with tf_compat.name_scope(name):
        global_step = tf_compat.cast(global_step, tf_compat.int64)
        milestone_steps = tf_compat.constant(
            [mile + start_step for mile in milestone_steps],
            dtype=tf_compat.int64,
            name="milestone_steps",
        )
        start_step = tf_compat.constant(
            start_step, dtype=tf_compat.int64, name="start_step"
        )
        init_lr = tf_compat.constant(init_lr, dtype=tf_compat.float32, name="init_lr")
        gamma = tf_compat.constant(gamma, dtype=tf_compat.float32, name="gamma")
        before = tf_compat.less(global_step, start_step, name="before")

        def _calc_lr():
            less = tf_compat.cast(
                tf_compat.greater_equal(global_step, milestone_steps), tf_compat.int64
            )
            updates = tf_compat.reduce_sum(less)
            mult_g = tf_compat.pow(gamma, tf_compat.cast(updates, tf_compat.float32))

            return tf_compat.multiply(init_lr, mult_g)

        learning_rate = tf_compat.cond(
            before, lambda: init_lr, _calc_lr, name="learning_rate"
        )

    return learning_rate
