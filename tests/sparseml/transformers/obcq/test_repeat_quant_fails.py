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

from parameterized import parameterized_class
from tests.testing_utils import parse_params, requires_torch


CONFIGS_DIRECTORY = "tests/sparseml/transformers/obcq/obcq_configs/repeat_quants"


@requires_torch
@pytest.mark.integration
@parameterized_class(parse_params(CONFIGS_DIRECTORY))
class TestRepeatQuants(unittest.TestCase):
    model = None
    first_recipe = None
    second_recipe = None
    dataset = None

    def setUp(self):
        import torch

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.output = "./oneshot_output"
        self.output_first = Path(self.output) / "test_1"
        self.output_second = Path(self.output) / "test_2"

    def test_fail_on_repeated_quant(self):
        import sparseml.core.session as session_manager
        from sparseml.transformers import oneshot

        oneshot(
            model=self.model,
            dataset=self.dataset,
            num_calibration_samples=4,
            oneshot_device=self.device,
            recipe=self.first_recipe,
            output_dir=self.output_first,
            clear_sparse_session=False,
        )

        session = session_manager.active_session()
        session.reset()

        # When trying to re-quantize with the second recipe, we should error out
        # to avoid nested quantizations
        with pytest.raises(RuntimeError):
            oneshot(
                model=self.output_first,
                dataset=self.dataset,
                num_calibration_samples=4,
                oneshot_device=self.device,
                recipe=self.second_recipe,
            )

    def tearDown(self):
        shutil.rmtree(self.output)
