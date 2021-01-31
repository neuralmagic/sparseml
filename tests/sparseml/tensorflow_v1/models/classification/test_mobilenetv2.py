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

from sparseml.tensorflow_v1.models import ModelRegistry, mobilenet_v2
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
        ("mobilenetv2", False, True, mobilenet_v2),
        ("mobilenetv2", True, False, mobilenet_v2),
        ("mobilenetv2", "base", False, mobilenet_v2),
    ],
)
def test_mobilenets_v2(
    key: str, pretrained: Union[bool, str], test_input: bool, const: Callable
):
    # test out the stand alone constructor
    with tf_compat.Graph().as_default():
        inputs = tf_compat.placeholder(
            tf_compat.float32, [None, 224, 224, 3], name="inputs"
        )
        logits = const(inputs, training=False)

        if test_input:
            with tf_compat.Session() as sess:
                sess.run(tf_compat.global_variables_initializer())
                out = sess.run(
                    logits, feed_dict={inputs: numpy.random.random((1, 224, 224, 3))}
                )
                assert out.sum() != 0

    # test out the registry
    with tf_compat.Graph().as_default():
        inputs = tf_compat.placeholder(
            tf_compat.float32, [None, 224, 224, 3], name="inputs"
        )
        logits = ModelRegistry.create(key, inputs, training=False)

        with tf_compat.Session() as sess:
            if test_input:
                sess.run(tf_compat.global_variables_initializer())
                out = sess.run(
                    logits, feed_dict={inputs: numpy.random.random((1, 224, 224, 3))}
                )
                assert out.sum() != 0

            if pretrained:
                ModelRegistry.load_pretrained(key, pretrained)

                if test_input:
                    out = sess.run(
                        logits,
                        feed_dict={inputs: numpy.random.random((1, 224, 224, 3))},
                    )
                    assert out.sum() != 0
