import pytest

from typing import Union, Callable
import numpy

from neuralmagicML.tensorflow.utils import tf_compat
from neuralmagicML.tensorflow.models import (
    ModelRegistry,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)


@pytest.mark.parametrize(
    "key,pretrained,test_input,const",
    [
        ("resnet18", False, True, resnet18),
        ("resnet18", True, True, resnet18),
        ("resnet18", "base", True, resnet18),
        ("resnet34", False, True, resnet34),
        ("resnet34", True, True, resnet34),
        ("resnet34", "base", True, resnet34),
        ("resnet50", False, True, resnet50),
        ("resnet50", True, False, resnet50),
        ("resnet50", "base", False, resnet50),
        ("resnet50", "recal", False, resnet50),
        ("resnet50", "recal-perf", False, resnet50),
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
    # test out the stand alone constructor
    with tf_compat.Graph().as_default() as graph:
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
    with tf_compat.Graph().as_default() as graph:
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
