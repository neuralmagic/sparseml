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

import shutil
import unittest

import pytest

from tests.testing_utils import requires_torch


@pytest.mark.integration
@requires_torch
class TestFakeQuantWrapper(unittest.TestCase):
    def setUp(self):
        import torch

        self.output = "./oneshot_output"
        self.model = "roneneldan/TinyStories-1M"
        self.dataset = "open_platypus"
        self.precision = "bfloat16"  # unsupported by native FakeQuantize
        self.device = (
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )  # unsupported by native FakeQuantize

        self.recipe = """
        first_stage:
            quant_modifiers:
                LegacyQuantizationModifier:
                    ignore:
                        - Embedding
                    scheme_overrides:
                        LayerNorm:
                            input_activations: null
                            output_activations: null
        """

    def test_fake_quant_wrapper(self):
        from sparseml.transformers import oneshot

        oneshot(
            model=self.model,
            dataset=self.dataset,
            output_dir=self.output,
            overwrite_output_dir=True,
            precision=self.precision,
            recipe=self.recipe,
            oneshot_device=self.device,
            num_calibration_samples=9,
        )

    def tearDown(self):
        shutil.rmtree(self.output)
