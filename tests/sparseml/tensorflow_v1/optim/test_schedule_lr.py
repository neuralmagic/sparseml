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

import os
from typing import List

import pytest

from sparseml.tensorflow_v1.optim import multi_step_lr_schedule, step_lr_schedule
from sparseml.tensorflow_v1.utils import tf_compat


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False),
    reason="Skipping tensorflow_v1 tests",
)
@pytest.mark.parametrize(
    "start_step,end_step,init_lr,step_size,gamma",
    [
        (0, 100, 0.01, 1, 0.9),
        (17, 100, 0.1, 15, 0.9),
        (0, 200, 0.01, 15, 0.95),
        (11, 200, 0.1, 7, 0.95),
        (11, -1, 0.1, 3, 0.95),
    ],
)
def test_step_lr_schedule(
    start_step: int, end_step: int, init_lr: float, step_size: int, gamma: float
):
    with tf_compat.Graph().as_default():
        global_step = tf_compat.placeholder(dtype=tf_compat.int64, shape=[])
        learning_rate = step_lr_schedule(
            global_step, start_step, end_step, step_size, init_lr, gamma
        )

        with tf_compat.Session() as sess:
            expected = init_lr

            for step in range(end_step + 10):
                measured = sess.run(learning_rate, feed_dict={global_step: step})

                if (
                    step - start_step
                ) % step_size == 0 and start_step < step <= end_step:
                    expected = expected * gamma

                assert abs(measured - expected) < 1e-5


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False),
    reason="Skipping tensorflow_v1 tests",
)
@pytest.mark.parametrize(
    "start_step,milestone_steps,init_lr,gamma",
    [
        (0, [5, 9, 13], 0.01, 0.9),
        (13, [9, 13], 0.1, 0.9),
        (0, [5, 9, 13], 0.01, 0.95),
        (17, [9, 13], 0.1, 0.9),
    ],
)
def test_multi_step_lr_schedule(
    start_step: int, milestone_steps: List[int], init_lr: float, gamma: float
):
    with tf_compat.Graph().as_default():
        global_step = tf_compat.placeholder(dtype=tf_compat.int64, shape=[])
        learning_rate = multi_step_lr_schedule(
            global_step, start_step, milestone_steps, init_lr, gamma
        )

        with tf_compat.Session() as sess:
            for step in range(start_step + milestone_steps[-1] + 10):
                measured = sess.run(learning_rate, feed_dict={global_step: step})

                gammas = sum(
                    [1 for mile in milestone_steps if step >= mile + start_step]
                )
                expected = init_lr * gamma**gammas

                assert abs(measured - expected) < 1e-5
