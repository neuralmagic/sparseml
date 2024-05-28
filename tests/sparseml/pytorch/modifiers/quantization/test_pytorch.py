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

import unittest

import pytest

from parameterized import parameterized
from sparseml.core import State
from sparseml.core.event import Event, EventType
from sparseml.core.factory import ModifierFactory
from sparseml.core.framework import Framework
from sparseml.modifiers.quantization_legacy.pytorch import (
    LegacyQuantizationModifierPyTorch,
)
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
from tests.testing_utils import requires_torch


@pytest.mark.unit
@requires_torch
class TestQuantizationRegistered(unittest.TestCase):
    def setUp(self):
        setup_modifier_factory()
        self.kwargs = dict(index=0, group="quantization", start=2.0, end=-1.0)

    def test_quantization_registered(self):
        quant_obj = ModifierFactory.create(
            type_="LegacyQuantizationModifier",
            framework=Framework.pytorch,
            allow_experimental=False,
            allow_registered=True,
            **self.kwargs,
        )

        self.assertIsInstance(quant_obj, LegacyQuantizationModifierPyTorch)


@pytest.mark.unit
@requires_torch
class TestQuantizationOneShot(unittest.TestCase):
    def setUp(self):
        scheme = dict(
            input_activations=dict(num_bits=8, symmetric=True),
            weights=dict(num_bits=4, symmetric=False, strategy="channel"),
        )
        self.kwargs = dict(scheme=scheme)

    @parameterized.expand([[ConvNet], [LinearNet]])
    def test_quantization_oneshot(self, model_class):
        model = model_class()
        state = State(framework=Framework.pytorch, start_event=Event())
        state.update(model=model, start=-1)

        modifier = LegacyQuantizationModifierPyTorch(**self.kwargs)

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


@pytest.mark.unit
@requires_torch
class TestQuantizationTraining(unittest.TestCase):
    def setUp(self):
        self.start_epoch = 2

        self.kwargs = dict(
            start=self.start_epoch,
            scheme=dict(
                input_activations=dict(num_bits=8, symmetric=True),
                weights=dict(num_bits=4, symmetric=False),
            ),
        )

    @parameterized.expand([[ConvNet], [LinearNet]])
    def test_quantization_training(self, model_class):
        model = model_class()

        modifier = LegacyQuantizationModifierPyTorch(**self.kwargs)

        testing_harness = LifecyleTestingHarness(model=model)
        modifier.initialize(testing_harness.get_state())
        assert not modifier.qat_enabled_

        testing_harness.trigger_modifier_for_epochs(modifier, self.start_epoch)
        assert not modifier.qat_enabled_
        testing_harness.trigger_modifier_for_epochs(modifier, self.start_epoch + 1)
        _test_qat_applied(modifier, model)

        modifier.finalize(testing_harness.get_state())
        assert modifier.quantization_observer_disabled_


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
