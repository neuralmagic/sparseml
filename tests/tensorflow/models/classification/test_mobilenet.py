import os
import pytest

from typing import Union, Callable
import numpy

from sparseml.tensorflow_v1.utils import tf_compat
from sparseml.tensorflow_v1.models import ModelRegistry, mobilenet


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False), reason="Skipping tensorflow_v1 tests",
)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_MODEL_TESTS", False), reason="Skipping model tests",
)
@pytest.mark.parametrize(
    "key,pretrained,test_input,const",
    [
        ("mobilenet", False, True, mobilenet),
        ("mobilenet", True, False, mobilenet),
        ("mobilenet", "base", False, mobilenet),
    ],
)
def test_mobilenets(
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
