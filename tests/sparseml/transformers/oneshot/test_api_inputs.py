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
from tests.sparseml.transformers.oneshot.dataset_processing import get_data_utils
from tests.testing_utils import parse_params, requires_torch


CONFIGS_DIRECTORY = "tests/sparseml/transformers/oneshot/oneshot_configs"

# TODO: Seems better to mark test type (smoke, sanity, regression) as a marker as
# opposed to using a field in the config file?


@pytest.mark.smoke
@pytest.mark.integration
@requires_torch
@parameterized_class(parse_params(CONFIGS_DIRECTORY))
class TestOneShotInputs(unittest.TestCase):
    model = None
    dataset = None
    recipe = None
    dataset_config_name = None
    tokenize = None

    def setUp(self):
        from sparseml.transformers import (
            SparseAutoModelForCausalLM,
            SparseAutoTokenizer,
        )

        self.tokenizer = SparseAutoTokenizer.from_pretrained(self.model)
        self.model = SparseAutoModelForCausalLM.from_pretrained(self.model)
        self.output = "./oneshot_output"
        self.kwargs = {"dataset_config_name": self.dataset_config_name}

        data_utils = get_data_utils(self.dataset)

        def wrapped_preprocess_func(sample):
            preprocess_func = data_utils.get("preprocess")
            return self.tokenizer(
                preprocess_func(sample), padding=False, max_length=512, truncation=True
            )

        # If `tokenize` is set to True, use the appropriate preprocessing function
        # and set self.tokenizer = None. Updates the self.dataset field from the string
        # to the loaded dataset.
        if self.tokenize:
            loaded_dataset = data_utils.get("dataload")()
            self.dataset = loaded_dataset.map(
                wrapped_preprocess_func,
                remove_columns=data_utils.get("remove_columns"),
            )
            self.tokenizer = None

    def test_one_shot_inputs(self):
        from sparseml.transformers import oneshot

        oneshot(
            model=self.model,
            tokenizer=self.tokenizer,
            dataset=self.dataset,
            recipe=self.recipe,
            output_dir=self.output,
            num_calibration_samples=10,
            pad_to_max_length=False,
            **self.kwargs,
        )

    def tearDown(self):
        shutil.rmtree(self.output)
