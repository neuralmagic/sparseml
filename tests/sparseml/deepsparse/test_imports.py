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


def test_imports():
    # flake8: noqa
    from sparseml.deepsparse import (
        check_deepsparse_install,
        deepsparse,
        deepsparse_err,
        detect_framework,
        framework_info,
        is_supported,
        require_deepsparse,
        sparsification_info,
    )
