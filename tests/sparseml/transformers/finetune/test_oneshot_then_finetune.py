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
import shutil
import unittest
from pathlib import Path

import pytest

from tests.testing_utils import requires_torch


@pytest.mark.integration
@requires_torch
@pytest.mark.skipif(
    "CADENCE" in os.environ
    and (os.environ["CADENCE"] == "weekly" or os.environ["CADENCE"] == "nightly"),
    reason="Don't run for weekly and nightly tests as those use multi gpu "
    "runners and this test fails when ngpu>1",
)
class TestOneshotThenFinetune(unittest.TestCase):
    def setUp(self):
        self.output = Path("./finetune_output")

    def test_oneshot_then_finetune(self):
        import sparseml
        from sparseml.transformers import SparseAutoModelForCausalLM, oneshot, train

        recipe_str = "tests/sparseml/transformers/obcq/recipes/test_tiny2.yaml"
        model = SparseAutoModelForCausalLM.from_pretrained(
            "Xenova/llama2.c-stories15M", device_map="auto"
        )
        dataset = "open_platypus"
        concatenate_data = False
        num_calibration_samples = 64
        output_dir = self.output / "oneshot_out"
        splits = {"calibration": "train[:10%]"}

        with sparseml.create_session():
            oneshot(
                model=model,
                dataset=dataset,
                output_dir=output_dir,
                num_calibration_samples=num_calibration_samples,
                recipe=recipe_str,
                concatenate_data=concatenate_data,
                splits=splits,
            )

        recipe_str = "tests/sparseml/transformers/finetune/test_finetune_recipe.yaml"
        model = SparseAutoModelForCausalLM.from_pretrained(
            self.output / "oneshot_out", device_map="auto"
        )
        distill_teacher = SparseAutoModelForCausalLM.from_pretrained(
            "Xenova/llama2.c-stories15M", device_map="auto"
        )
        dataset = "open_platypus"
        concatenate_data = False
        output_dir = self.output / "finetune_out"
        splits = "train[:50%]"
        max_steps = 50

        with sparseml.create_session():
            train(
                model=model,
                distill_teacher=distill_teacher,
                dataset=dataset,
                output_dir=output_dir,
                num_calibration_samples=num_calibration_samples,
                recipe=recipe_str,
                concatenate_data=concatenate_data,
                splits=splits,
                max_steps=max_steps,
            )

    def tearDown(self):
        shutil.rmtree(self.output)
