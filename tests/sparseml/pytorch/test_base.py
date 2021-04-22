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

from sparseml.pytorch.base import (
    check_torch_install,
    check_torchvision_install,
    require_torch,
    require_torchvision,
    torch,
    torch_err,
    torchvision,
    torchvision_err,
)


def test_torch():
    assert torch
    assert not torch_err
    assert torchvision
    assert not torchvision_err


def test_check_torch_install():
    assert check_torch_install()

    assert not check_torch_install(min_version="10.0.0", raise_on_error=False)
    with pytest.raises(ImportError):
        check_torch_install(min_version="10.0.0")

    assert not check_torch_install(max_version="0.0.1", raise_on_error=False)
    with pytest.raises(ImportError):
        check_torch_install(max_version="0.0.1")


def test_check_torchvision_install():
    assert check_torchvision_install()

    assert not check_torchvision_install(min_version="10.0.0", raise_on_error=False)
    with pytest.raises(ImportError):
        check_torchvision_install(min_version="10.0.0")

    assert not check_torchvision_install(max_version="0.0.1", raise_on_error=False)
    with pytest.raises(ImportError):
        check_torchvision_install(max_version="0.0.1")


def test_require_torch():
    @require_torch()
    def _func_one(arg1, arg2, arg3):
        assert arg1
        assert arg2
        assert arg3

    _func_one(arg1=1, arg2=2, arg3=3)

    @require_torch(min_version="10.0.0")
    def _func_two():
        pass

    with pytest.raises(ImportError):
        _func_two()


def test_require_torchvision():
    @require_torchvision()
    def _func_one(arg1, arg2, arg3):
        assert arg1
        assert arg2
        assert arg3

    _func_one(arg1=1, arg2=2, arg3=3)

    @require_torchvision(min_version="10.0.0")
    def _func_two():
        pass

    with pytest.raises(ImportError):
        _func_two()
