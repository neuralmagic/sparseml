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

import pytest

from sparseml.keras.base import (
    check_keras2onnx_install,
    check_keras_install,
    is_native_keras,
    keras,
    keras2onnx,
    keras2onnx_err,
    keras_err,
    require_keras,
    require_keras2onnx,
    tensorflow,
    tensorflow_err,
)


def test_keras():
    assert keras
    assert not keras_err
    assert tensorflow
    assert not tensorflow_err
    assert keras2onnx
    assert not keras2onnx_err
    assert is_native_keras if keras != tensorflow.keras else not is_native_keras


def test_check_keras_install():
    assert check_keras_install()

    assert not check_keras_install(min_tf_version="10.0.0", raise_on_error=False)
    with pytest.raises(ImportError):
        check_keras_install(min_tf_version="10.0.0")

    assert not check_keras_install(max_tf_version="0.0.1", raise_on_error=False)
    with pytest.raises(ImportError):
        check_keras_install(max_tf_version="0.0.1")


def test_check_keras2onnx_install():
    assert check_keras2onnx_install()

    assert not check_keras2onnx_install(min_version="10.0.0", raise_on_error=False)
    with pytest.raises(ImportError):
        check_keras2onnx_install(min_version="10.0.0")

    assert not check_keras2onnx_install(max_version="0.0.1", raise_on_error=False)
    with pytest.raises(ImportError):
        check_keras2onnx_install(max_version="0.0.1")


def test_require_keras():
    @require_keras()
    def _func_one(arg1, arg2, arg3):
        assert arg1
        assert arg2
        assert arg3

    _func_one(arg1=1, arg2=2, arg3=3)

    @require_keras(min_tf_version="10.0.0")
    def _func_two():
        pass

    with pytest.raises(ImportError):
        _func_two()


def test_require_keras2onnx():
    @require_keras2onnx()
    def _func_one(arg1, arg2, arg3):
        assert arg1
        assert arg2
        assert arg3

    _func_one(arg1=1, arg2=2, arg3=3)

    @require_keras2onnx(min_version="10.0.0")
    def _func_two():
        pass

    with pytest.raises(ImportError):
        _func_two()
