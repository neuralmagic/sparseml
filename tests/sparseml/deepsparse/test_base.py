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

from sparseml.deepsparse.base import (
    check_deepsparse_install,
    deepsparse,
    deepsparse_err,
    require_deepsparse,
)


def test_deepsparse():
    assert deepsparse
    assert not deepsparse_err


def test_check_deepsparse_install():
    assert check_deepsparse_install()

    assert not check_deepsparse_install(min_version="10.0.0", raise_on_error=False)
    with pytest.raises(ImportError):
        check_deepsparse_install(min_version="10.0.0")

    assert not check_deepsparse_install(max_version="0.0.1", raise_on_error=False)
    with pytest.raises(ImportError):
        check_deepsparse_install(max_version="0.0.1")


def test_require_deepsparse():
    @require_deepsparse()
    def _func_one(arg1, arg2, arg3):
        assert arg1
        assert arg2
        assert arg3

    _func_one(arg1=1, arg2=2, arg3=3)

    @require_deepsparse(min_version="10.0.0")
    def _func_two():
        pass

    with pytest.raises(ImportError):
        _func_two()
