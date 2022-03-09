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
from typing import Callable, Union

import numpy
import pytest

from sparseml.tensorflow_v1.models import (
    ModelRegistry,
    resnet18,
    resnet20,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)
from sparseml.tensorflow_v1.utils import tf_compat


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False),
    reason="Skipping tensorflow_v1 tests",
)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_MODEL_TESTS", False),
    reason="Skipping model tests",
)
@pytest.mark.parametrize(
    "key,pretrained,test_input,const",
    [
        ("resnet18", False, True, resnet18),
        ("resnet18", True, True, resnet18),
        ("resnet18", "base", True, resnet18),
        ("resnet20", False, True, resnet20),
        ("resnet34", False, True, resnet34),
        ("resnet34", True, True, resnet34),
        ("resnet34", "base", True, resnet34),
        ("resnet50", False, True, resnet50),
        ("resnet50", True, False, resnet50),
        ("resnet50", "base", False, resnet50),
        ("resnet50", "optim", False, resnet50),
        ("resnet50", "optim-perf", False, resnet50),
        ("resnet101", False, True, resnet101),
        ("resnet101", True, False, resnet101),
        ("resnet101", "base", False, resnet101),
        ("resnet152", False, True, resnet152),
        ("resnet152", True, False, resnet152),
        ("resnet152", "base", False, resnet152),
    ],
)
def test_resnets(
    key: str, pretrained: Union[bool, str], test_input: bool, const: Callable
):
    input_shape = ModelRegistry.input_shape(key)
    # test out the stand alone constructor
    with tf_compat.Graph().as_default():
        inputs = tf_compat.placeholder(
            tf_compat.float32, [None, *input_shape], name="inputs"
        )
        logits = const(inputs, training=False)

        if test_input:
            with tf_compat.Session() as sess:
                sess.run(tf_compat.global_variables_initializer())
                out = sess.run(
                    logits, feed_dict={inputs: numpy.random.random((1, *input_shape))}
                )
                assert out.sum() != 0

    # test out the registry
    with tf_compat.Graph().as_default():
        inputs = tf_compat.placeholder(
            tf_compat.float32, [None, *input_shape], name="inputs"
        )
        logits = ModelRegistry.create(key, inputs, training=False)

        with tf_compat.Session() as sess:
            if test_input:
                sess.run(tf_compat.global_variables_initializer())
                out = sess.run(
                    logits, feed_dict={inputs: numpy.random.random((1, *input_shape))}
                )
                assert out.sum() != 0

            if pretrained:
                ModelRegistry.load_pretrained(key, pretrained)

                if test_input:
                    out = sess.run(
                        logits,
                        feed_dict={inputs: numpy.random.random((1, *input_shape))},
                    )
                    assert out.sum() != 0
