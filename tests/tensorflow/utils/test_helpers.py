import pytest

import os


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False), reason="Skipping tensorflow tests",
)
def test_tf_compat():
    from neuralmagicML.tensorflow.utils import tf_compat

    assert tf_compat
    assert tf_compat.Graph


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False), reason="Skipping tensorflow tests",
)
def test_tf_compat_div():
    from neuralmagicML.tensorflow.utils import tf_compat_div

    assert tf_compat_div
