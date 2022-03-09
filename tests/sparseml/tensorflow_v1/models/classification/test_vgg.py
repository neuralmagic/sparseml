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
    vgg11,
    vgg11bn,
    vgg13,
    vgg13bn,
    vgg16,
    vgg16bn,
    vgg19,
    vgg19bn,
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
        ("vgg11", False, True, vgg11),
        ("vgg11", True, False, vgg11),
        ("vgg11bn", False, True, vgg11bn),
        ("vgg11bn", True, False, vgg11bn),
        ("vgg13", False, True, vgg13),
        ("vgg13", True, False, vgg13),
        ("vgg13bn", False, True, vgg13bn),
        ("vgg13bn", True, False, vgg13bn),
        ("vgg16", False, True, vgg16),
        ("vgg16", True, False, vgg16),
        ("vgg16", "base", False, vgg16),
        ("vgg16", "optim", False, vgg16),
        ("vgg16", "optim-perf", False, vgg16),
        ("vgg16bn", False, True, vgg16bn),
        ("vgg16bn", True, False, vgg16bn),
        ("vgg19", False, True, vgg19),
        ("vgg19", True, False, vgg19),
        ("vgg19bn", False, True, vgg19bn),
        ("vgg19bn", True, False, vgg19bn),
    ],
)
def test_vggs(
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
