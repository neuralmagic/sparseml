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

import os

import pytest

from sparseml.pytorch.sparsification.quantization.modifier_quantization import (
    QuantizationModifier,
)
from sparseml.pytorch.sparsification.quantization.quantize import QuantizationScheme


try:
    from torch import quantization as torch_quantization
except Exception:
    torch_quantization = None


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_QUANT_TESTS", False),
    reason="Skipping pytorch torch quantization tests",
)
@pytest.mark.skipif(
    torch_quantization is None,
    reason="torch quantization not available",
)
def test_quantization_modifier_yaml():
    start_epoch = 0.0
    default_scheme = dict(
        input_activations=dict(num_bits=8, symmetric=True),
        weights=dict(num_bits=6, symmetric=False),
    )
    yaml_str = f"""
        !QuantizationModifier
            start_epoch: {start_epoch}
            default_scheme: {default_scheme}
        """
    yaml_modifier = QuantizationModifier.load_obj(
        yaml_str
    )  # type: QuantizationModifier
    serialized_modifier = QuantizationModifier.load_obj(
        str(yaml_modifier)
    )  # type: QuantizationModifier
    obj_modifier = QuantizationModifier(
        start_epoch=start_epoch,
        default_scheme=default_scheme,
    )

    assert isinstance(yaml_modifier, QuantizationModifier)
    assert (
        yaml_modifier.start_epoch
        == serialized_modifier.start_epoch
        == obj_modifier.start_epoch
    )
    assert (
        yaml_modifier.default_scheme
        == serialized_modifier.default_scheme
        == obj_modifier.default_scheme
    )
    assert isinstance(yaml_modifier.default_scheme, QuantizationScheme)
    assert isinstance(serialized_modifier.default_scheme, QuantizationScheme)
    assert isinstance(obj_modifier.default_scheme, QuantizationScheme)
