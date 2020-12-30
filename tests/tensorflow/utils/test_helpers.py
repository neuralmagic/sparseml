import pytest

import os


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False), reason="Skipping tensorflow_v1 tests",
)
def test_tf_compat():
    from sparseml.tensorflow_v1.utils import tf_compat

    assert tf_compat
    assert tf_compat.Graph


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False), reason="Skipping tensorflow_v1 tests",
)
def test_tf_compat_div():
    from sparseml.tensorflow_v1.utils import tf_compat_div

    assert tf_compat_div
