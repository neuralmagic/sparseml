def test_tf_compat():
    from neuralmagicML.tensorflow.utils import tf_compat

    assert tf_compat
    assert tf_compat.Graph


def test_tf_compat_div():
    from neuralmagicML.tensorflow.utils import tf_compat_div

    assert tf_compat_div
