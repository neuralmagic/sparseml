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

from sparseml.modifiers.obcq.pytorch import SparseGPTModifierPyTorch
from sparseml.modifiers.quantization import QuantizationModifier
from sparseml.modifiers.quantization.pytorch import QuantizationModifierPyTorch
from tests.sparseml.modifiers.conf import LifecyleTestingHarness, setup_modifier_factory
from tests.sparseml.pytorch.helpers import LinearNet


def test_create_default_quant_modifier():
    setup_modifier_factory()
    kwargs = dict(sparsity=0.5, block_size=128, quantize=True)

    modifier = SparseGPTModifierPyTorch(**kwargs)
    assert modifier.quantization_modifier_ is None

    testing_harness = LifecyleTestingHarness(model=LinearNet())
    modifier.pre_initialize_structure(testing_harness.get_state())
    assert modifier.quantize
    assert isinstance(modifier.quantization_modifier_, QuantizationModifier)

    should_be_default_quant_scheme = modifier.quantization_modifier_.scheme
    assert should_be_default_quant_scheme.input_activations.num_bits == 8
    assert not should_be_default_quant_scheme.input_activations.symmetric
    assert should_be_default_quant_scheme.weights.num_bits == 8
    assert should_be_default_quant_scheme.weights.symmetric


def test_set_quant_if_modifer_already_exists():
    setup_modifier_factory()

    model = LinearNet()
    kwargs = dict(
        scheme=dict(
            input_activations=dict(num_bits=8, symmetric=True),
            weights=dict(num_bits=4, symmetric=False),
        ),
    )

    modifier = QuantizationModifierPyTorch(**kwargs)
    testing_harness = LifecyleTestingHarness(model=model)

    assert not testing_harness.get_state().model.qat_active()
    modifier.initialize(testing_harness.get_state())
    assert testing_harness.get_state().model.qat_active()

    kwargs = dict(sparsity=0.5, block_size=128, quantize=False)
    modifier = SparseGPTModifierPyTorch(**kwargs)
    assert not modifier.quantize
    modifier.pre_initialize_structure(testing_harness.get_state())

    # quantization modifier not owned by SparseGPT
    assert modifier.quantization_modifier_ is None

    # since quantization modifier is already applied, quantization must be set in OBCQ
    assert modifier.quantize


def test_set_quant_in_sparsegpt():
    setup_modifier_factory()

    quant_kwargs = {
        "scheme": {
            "input_activations": {
                "num_bits": 8,
                "symmetric": False,
                "strategy": "tensor",
                "kwargs": {},
            },
            "weights": {
                "num_bits": 4,
                "symmetric": True,
                "strategy": "channel",
                "kwargs": {},
            },
        }
    }
    quant_config = {"QuantizationModifier": quant_kwargs}

    kwargs = dict(sparsity=0.5, block_size=128, quantize=quant_config)

    modifier = SparseGPTModifierPyTorch(**kwargs)
    assert modifier.quantization_modifier_ is None

    testing_harness = LifecyleTestingHarness(model=LinearNet())
    modifier.pre_initialize_structure(testing_harness.get_state())
    assert modifier.quantize
    assert isinstance(modifier.quantization_modifier_, QuantizationModifier)

    dict_scheme = dict(modifier.quantization_modifier_.scheme)
    assert dict(dict_scheme["weights"]) == quant_kwargs["scheme"]["weights"]
    assert (
        dict(dict_scheme["input_activations"])
        == quant_kwargs["scheme"]["input_activations"]
    )
