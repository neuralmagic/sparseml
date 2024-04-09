import unittest
from pathlib import Path
from parameterized import parameterized_class
from tests.testing_utils import parse_params, requires_torch

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
        from sparseml.transformers import SparseAutoModelForCausalLM, SparseAutoTokenizer
        self.tokenizer = SparseAutoTokenizer.from_pretrained(self.model)
        self.model = SparseAutoModelForCausalLM.from_pretrained(self.model)
        
        self.output = "./oneshot_output"

    def test_one_shot_inputs(self):
        from sparseml.transformers import oneshot

        oneshot(
            model=self.model,
            tokenizer=self.tokenizer,
            dataset=self.dataset,
            recipe=self.recipe,
            output_dir=self.output,
            num_calibration_samples=10,
        )

    def tearDown(self):
        shutil.rmtree(self.output)
