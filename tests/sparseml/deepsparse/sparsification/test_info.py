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

from sparseml.base import Framework
from sparseml.deepsparse.sparsification import sparsification_info
from sparseml.sparsification import sparsification_info as base_sparsification_info


def test_sparsification_info():
    base_info = base_sparsification_info(Framework.deepsparse)
    info = sparsification_info()
    assert base_info == info

    assert len(info.modifiers) == 0
