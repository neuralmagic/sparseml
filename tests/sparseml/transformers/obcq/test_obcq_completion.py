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

from parameterized import parameterized_class
from tests.testing_utils import parse_params, requires_gpu, requires_torch


CONFIGS_DIRECTORY = "tests/sparseml/transformers/obcq/obcq_configs/completion"
GPU_CONFIGS_DIRECTORY = "tests/sparseml/transformers/obcq/obcq_configs/completion/gpu"


@requires_torch
@pytest.mark.integration
@parameterized_class(parse_params(CONFIGS_DIRECTORY))
class TestOBCQCompletion(unittest.TestCase):
    """
    Test for oneshot for quantization and quantization + sparsity. Sparsity-only tests
    can be found under `test_obcq_sparsity.py`
    """

    model = None
    dataset = None
    recipe = None
    sparsity = None

    def setUp(self):
        import torch

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.output = "./oneshot_output"

    def test_oneshot_completion(self):
        from sparseml.transformers import oneshot

        oneshot(
            model=self.model,
            dataset=self.dataset,
            oneshot_device=self.device,
            recipe=self.recipe,
            max_seq_length=128,
            num_calibration_samples=32,
            pad_to_max_length=False,
            output_dir=self.output,
        )

    def tearDown(self):
        shutil.rmtree(self.output)


@requires_torch
@requires_gpu
@pytest.mark.integration
@parameterized_class(parse_params(GPU_CONFIGS_DIRECTORY))
class TestOBCQCompletionGPU(unittest.TestCase):
    """
    Test for oneshot for quantization and quantization + sparsity. Sparsity-only tests
    can be found under `test_obcq_sparsity.py`
    """

    model = None
    dataset = None
    recipe = None
    sparsity = None
    device = None

    def setUp(self):
        from sparseml.transformers import SparseAutoModelForCausalLM

        self.output = "./oneshot_output"

        if "zoo:" in self.model:
            self.model = SparseAutoModelForCausalLM.from_pretrained(
                self.model, device_map=self.device
            )

    def test_oneshot_completion(self):
        from sparseml.transformers import oneshot

        oneshot(
            model=self.model,
            dataset=self.dataset,
            oneshot_device=self.device,
            recipe=self.recipe,
            max_seq_length=128,
            num_calibration_samples=32,
            pad_to_max_length=False,
            output_dir=self.output,
        )

    def tearDown(self):
        shutil.rmtree(self.output)
