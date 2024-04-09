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
from subprocess import PIPE, STDOUT, run

import pytest

from parameterized import parameterized_class
from tests.testing_utils import parse_params, requires_torch


CONFIGS_DIRECTORY = "tests/sparseml/pytorch/oneshot/oneshot_configs"


@pytest.mark.smoke
@pytest.mark.integration
@requires_torch
@parameterized_class(parse_params(CONFIGS_DIRECTORY))
class TestOneShotCli(unittest.TestCase):
    model = None
    dataset = None
    recipe = None

    def setUp(self):
        self.output = "./oneshot_output"

    def test_one_shot_cli(self):
        cmd = [
            "sparseml.transformers.text_generation.oneshot",
            "--dataset",
            self.dataset,
            "--model",
            self.model,
            "--output_dir",
            self.output,
            "--recipe",
            self.recipe,
            "--num_calibration_samples",
            "10",
        ]

        res = run(cmd, stdout=PIPE, stderr=STDOUT, check=False, encoding="utf-8")
        self.assertEqual(res.returncode, 0)
        print(res.stdout)

    def tearDown(self):
        shutil.rmtree(self.output)
