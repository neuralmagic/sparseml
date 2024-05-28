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

from sparseml.core.event import Event
from sparseml.core.factory import ModifierFactory
from sparseml.core.framework import Framework
from sparseml.modifiers.quantization_legacy import LegacyQuantizationModifier
from tests.sparseml.modifiers.conf import setup_modifier_factory


@pytest.mark.unit
class TestQuantizationRegistered(unittest.TestCase):
    def setUp(self):
        setup_modifier_factory()
        self.kwargs = dict(index=0, group="quantization", start=2.0, end=-1.0)

    def test_quantization_registered(self):
        quant_obj = ModifierFactory.create(
            type_="LegacyQuantizationModifier",
            framework=Framework.general,
            allow_experimental=False,
            allow_registered=True,
            **self.kwargs,
        )

        self.assertIsInstance(quant_obj, LegacyQuantizationModifier)


@pytest.mark.unit
class TestEndEpochs(unittest.TestCase):
    def setUp(self):
        self.start = 0.0
        self.scheme = dict(
            input_activations=dict(num_bits=8, symmetric=True),
            weights=dict(num_bits=6, symmetric=False),
        )

    def test_end_epochs(self):
        disable_quant_epoch, freeze_bn_epoch = None, None
        obj_modifier = LegacyQuantizationModifier(
            start=self.start,
            scheme=self.scheme,
            disable_quantization_observer_epoch=disable_quant_epoch,
            freeze_bn_stats_epoch=freeze_bn_epoch,
        )

        self.assertEqual(obj_modifier.calculate_disable_observer_epoch(), -1)
        self.assertEqual(obj_modifier.calculate_freeze_bn_stats_epoch(), -1)

        for epoch in range(3):
            event = Event(steps_per_epoch=1, global_step=epoch)
            assert not obj_modifier.check_should_disable_observer(event)
            assert not obj_modifier.check_should_freeze_bn_stats(event)

        disable_quant_epoch, freeze_bn_epoch = 3.5, 5.0
        obj_modifier = LegacyQuantizationModifier(
            start=self.start,
            scheme=self.scheme,
            disable_quantization_observer_epoch=disable_quant_epoch,
            freeze_bn_stats_epoch=freeze_bn_epoch,
        )

        self.assertEqual(
            obj_modifier.calculate_disable_observer_epoch(), disable_quant_epoch
        )
        self.assertEqual(
            obj_modifier.calculate_freeze_bn_stats_epoch(), freeze_bn_epoch
        )

        for epoch in range(4):
            event = Event(steps_per_epoch=1, global_step=epoch)
            assert not obj_modifier.check_should_disable_observer(event)
            assert not obj_modifier.check_should_freeze_bn_stats(event)

        event = Event(steps_per_epoch=1, global_step=4)
        assert obj_modifier.check_should_disable_observer(event)
        assert not obj_modifier.check_should_freeze_bn_stats(event)

        for epoch in range(5, 8):
            event = Event(steps_per_epoch=1, global_step=epoch)
            assert obj_modifier.check_should_disable_observer(event)
            assert obj_modifier.check_should_freeze_bn_stats(event)
