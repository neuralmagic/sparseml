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
from pathlib import Path

import pytest

from tests.testing_utils import requires_torch


@pytest.mark.integration
@requires_torch
class TestOneshotWithModifierObject(unittest.TestCase):
    def setUp(self):
        self.output = Path("./finetune_output")

    def test_oneshot_with_modifier_object(self):
        import torch

        from sparseml.modifiers.obcq.base import SparseGPTModifier
        from sparseml.transformers import oneshot

        recipe_str = [
            SparseGPTModifier(sparsity=0.5, targets=[r"re:model.layers.\d+$"])
        ]
        model = "Xenova/llama2.c-stories15M"
        device = "cuda:0"
        if not torch.cuda.is_available():
            device = "cpu"
        dataset = "open_platypus"
        concatenate_data = False
        num_calibration_samples = 64
        output_dir = self.output / "oneshot_out"
        splits = {"calibration": "train[:10%]"}

        oneshot(
            model=model,
            dataset=dataset,
            output_dir=output_dir,
            num_calibration_samples=num_calibration_samples,
            recipe=recipe_str,
            concatenate_data=concatenate_data,
            splits=splits,
            oneshot_device=device,
        )

    def tearDown(self):
        shutil.rmtree(self.output)
