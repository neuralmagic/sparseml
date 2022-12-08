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
import torch
from packaging import version

from sparseml.pytorch.sparsification.quantization import (
    QuantizationArgs,
    QuantizationScheme,
)


@pytest.mark.parametrize(
    "quantization_args_dict,target_quantization_args",
    [
        (
            dict(num_bits=8, symmetric=False),
            QuantizationArgs(num_bits=8, symmetric=False),
        ),
        (
            dict(num_bits=4, symmetric=True),
            QuantizationArgs(num_bits=4, symmetric=True),
        ),
        (
            dict(num_bits=8, kwargs=dict(reduce_range=True)),
            QuantizationArgs(num_bits=8, kwargs=dict(reduce_range=True)),
        ),
    ],
)
def test_quantization_args_from_dict(quantization_args_dict, target_quantization_args):
    loaded_quantization_args = QuantizationArgs.parse_obj(quantization_args_dict)
    assert isinstance(loaded_quantization_args, QuantizationArgs)
    assert loaded_quantization_args == target_quantization_args


@pytest.mark.parametrize(
    "quantization_scheme_dict,target_quantization_scheme",
    [
        (
            dict(input_activations=None, weights=None, output_activations=None),
            QuantizationScheme(
                input_activations=None, weights=None, output_activations=None
            ),
        ),
        (
            dict(
                input_activations=dict(num_bits=8, symmetric=False),
                weights=dict(num_bits=4, symmetric=True),
                output_activations=dict(num_bits=8, kwargs=dict(reduce_range=True)),
            ),
            QuantizationScheme(
                input_activations=QuantizationArgs(num_bits=8, symmetric=False),
                weights=QuantizationArgs(num_bits=4, symmetric=True),
                output_activations=QuantizationArgs(
                    num_bits=8, kwargs=dict(reduce_range=True)
                ),
            ),
        ),
    ],
)
def test_quantization_scheme_from_dict(
    quantization_scheme_dict, target_quantization_scheme
):
    loaded_quantization_scheme = QuantizationScheme.parse_obj(quantization_scheme_dict)
    assert isinstance(loaded_quantization_scheme, QuantizationScheme)

    def _assert_none_or_is_quant_args(val):
        assert val is None or isinstance(val, QuantizationArgs)

    _assert_none_or_is_quant_args(loaded_quantization_scheme.input_activations)
    _assert_none_or_is_quant_args(loaded_quantization_scheme.weights)
    _assert_none_or_is_quant_args(loaded_quantization_scheme.output_activations)

    assert loaded_quantization_scheme == target_quantization_scheme


@pytest.mark.parametrize(
    "quantization_args,target_quant_min,target_quant_max",
    [
        (QuantizationArgs(num_bits=8, symmetric=False), -128, 127),
        (QuantizationArgs(num_bits=4, symmetric=True), -8, 7),
    ],
)
def test_quantization_args_get_observer(
    quantization_args, target_quant_min, target_quant_max
):
    observer = quantization_args.get_observer()
    assert hasattr(observer, "with_args")

    if quantization_args.num_bits == 8 and (
        version.parse(torch.__version__) >= version.parse("1.12.0")
    ):
        # quant min and max not set for default in later versions, no need to parse
        return

    assert observer.p.keywords["quant_min"] == target_quant_min
    assert observer.p.keywords["quant_max"] == target_quant_max


@pytest.mark.parametrize(
    "scheme_str,expected_scheme",
    [
        ("default", QuantizationScheme()),
        (
            "deepsparse",
            QuantizationScheme(
                input_activations=QuantizationArgs(num_bits=8, symmetric=False),
                weights=QuantizationArgs(num_bits=8, symmetric=True),
                output_activations=None,
                target_hardware="deepsparse",
            ),
        ),
        (
            "tensorrt",
            QuantizationScheme(
                input_activations=QuantizationArgs(num_bits=8, symmetric=True),
                weights=QuantizationArgs(num_bits=8, symmetric=True),
                output_activations=None,
                target_hardware="tensorrt",
            ),
        ),
        # adding to raise an issue if default scheme changes from deepsparse
        ("deepsparse", QuantizationScheme(target_hardware="deepsparse")),
    ],
)
def test_load_quantization_scheme_from_str(scheme_str, expected_scheme):
    loaded_scheme = QuantizationScheme.load(scheme_str)
    assert loaded_scheme == expected_scheme
