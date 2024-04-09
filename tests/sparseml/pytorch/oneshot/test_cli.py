import unittest
from pathlib import Path
from parameterized import parameterized_class
from tests.testing_utils import parse_params, requires_torch
from subprocess import PIPE, STDOUT, run

import pytest

import shutil

CONFIGS_DIRECTORY = "tests/sparseml/pytorch/oneshot/oneshot_configs"

@pytest.mark.smoke
@pytest.mark.integration
@requires_torch
@parameterized_class(parse_params(CONFIGS_DIRECTORY))
class TestOneShotInputs(unittest.TestCase):
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
            "10"
        ]

        res = run(cmd, stdout=PIPE, stderr=STDOUT, check=False, encoding="utf-8")
        self.assertEqual(res.returncode, 0)
        print(res.stdout)

    def tearDown(self):
        shutil.rmtree(self.output)
