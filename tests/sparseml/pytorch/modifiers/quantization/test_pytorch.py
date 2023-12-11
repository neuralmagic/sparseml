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

from sparseml.core import State
from sparseml.core.event import Event, EventType
from sparseml.core.factory import ModifierFactory
from sparseml.core.framework import Framework
from sparseml.modifiers.quantization.pytorch import QuantizationModifierPyTorch
from sparseml.pytorch.sparsification.quantization.quantize import (
    is_qat_helper_module,
    is_quantizable_module,
)
from tests.sparseml.modifiers.conf import LifecyleTestingHarness, setup_modifier_factory
from tests.sparseml.pytorch.helpers import ConvNet, LinearNet
from tests.sparseml.pytorch.sparsification.quantization.test_modifier_quantization import (  # noqa E501
    _match_submodule_name_or_type,
    _test_qat_wrapped_module,
    _test_quantized_module,
)


def _test_qat_applied(modifier, model):
    assert modifier.qat_enabled_

    for name, module in model.named_modules():
        if is_qat_helper_module(module):
            # skip helper modules
            continue

        is_target_submodule = not any(
            name.startswith(submodule_name) for submodule_name in modifier.ignore
        )
        is_included_module_type = any(
            module_type_name == module.__class__.__name__
            for module_type_name in modifier.scheme_overrides
        )
        is_quantizable = is_included_module_type or is_quantizable_module(
            module,
            exclude_module_types=modifier.ignore,
        )

        if is_target_submodule and is_quantizable:
            if getattr(module, "wrap_qat", False):
                _test_qat_wrapped_module(model, name)
            elif is_quantizable:
                # check each target module is quantized
                override_key = _match_submodule_name_or_type(
                    module,
                    name,
                    list(modifier.scheme_overrides.keys()),
                )
                _test_quantized_module(model, modifier, module, name, override_key)
        else:
            # check all non-target modules are not quantized
            assert not hasattr(module, "quantization_scheme")
            assert not hasattr(module, "qconfig")


def test_quantization_registered():
    setup_modifier_factory()

    kwargs = dict(index=0, group="quantization", start=2.0, end=-1.0)
    quant_obj = ModifierFactory.create(
        type_="QuantizationModifier",
        framework=Framework.pytorch,
        allow_experimental=False,
        allow_registered=True,
        **kwargs,
    )

    assert isinstance(quant_obj, QuantizationModifierPyTorch)


@pytest.mark.parametrize("model_class", [ConvNet, LinearNet])
def test_quantization_oneshot(model_class):
    model = model_class()
    state = State(framework=Framework.pytorch, start_event=Event())
    state.update(model=model, start=-1)

    scheme = dict(
        input_activations=dict(num_bits=8, symmetric=True),
        weights=dict(num_bits=4, symmetric=False, strategy="channel"),
    )
    kwargs = dict(scheme=scheme)

    modifier = QuantizationModifierPyTorch(**kwargs)

    modifier.initialize(state)

    # for one-shot, we set up quantization on initialization
    _test_qat_applied(modifier, model)

    # we shouldn't keep updating stats after one-shot
    assert modifier.quantization_observer_disabled_

    test_start_event = Event(type_=EventType.BATCH_START)
    test_end_event = Event(type_=EventType.BATCH_END)
    assert not modifier.should_start(test_start_event)
    assert not modifier.should_end(test_end_event)

    modifier.finalize(state)
    assert modifier.finalized


@pytest.mark.parametrize("model_class", [ConvNet, LinearNet])
def test_quantization_training(model_class):
    start_epoch = 2

    model = model_class()
    kwargs = dict(
        start=start_epoch,
        scheme=dict(
            input_activations=dict(num_bits=8, symmetric=True),
            weights=dict(num_bits=4, symmetric=False),
        ),
    )

    modifier = QuantizationModifierPyTorch(**kwargs)

    testing_harness = LifecyleTestingHarness(model=model)
    modifier.initialize(testing_harness.get_state())
    assert not modifier.qat_enabled_

    testing_harness.trigger_modifier_for_epochs(modifier, start_epoch)
    assert not modifier.qat_enabled_
    testing_harness.trigger_modifier_for_epochs(modifier, start_epoch + 1)
    _test_qat_applied(modifier, model)

    modifier.finalize(testing_harness.get_state())
    assert modifier.quantization_observer_disabled_
