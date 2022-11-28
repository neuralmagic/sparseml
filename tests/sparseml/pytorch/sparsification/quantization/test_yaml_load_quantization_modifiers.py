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

"""
The deprecated and latest QuantizationModifier objects share the same class name
with the yaml load process determining the correct object to load in order to maintain
backwards compatibility with existing model checkpoints and recipes. These tests
are in place to verify that the correct routing takes place
"""


import pytest

from sparseml.pytorch.sparsification import Modifier
from sparseml.pytorch.sparsification.quantization.legacy_modifier_quantization import (
    QuantizationModifier as LegacyQuantizationModifier,
)
from sparseml.pytorch.sparsification.quantization.modifier_quantization import (
    QuantizationModifier as LatestQuantizationModifier,
)


legacy_yaml = """
!QuantizationModifier
  start_epoch: 0.0
  submodules: [""]
"""

latest_yaml = """
!QuantizationModifier
  start_epoch: -1.0
"""


@pytest.mark.parametrize(
    "quant_modifier_yaml,is_legacy",
    [
        (legacy_yaml, True),
        (latest_yaml, False),
    ],
)
def test_yaml_load_quant_modifier_object_selection(quant_modifier_yaml, is_legacy):
    quant_modifier = Modifier.load_obj(quant_modifier_yaml)

    if is_legacy:
        assert isinstance(quant_modifier, LegacyQuantizationModifier)
    else:
        assert isinstance(quant_modifier, LatestQuantizationModifier)
