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


CONFIGS_DIRECTORY = "tests/sparseml/transformers/obcq/obcq_configs/separate_quants"


@requires_torch
@pytest.mark.integration
@parameterized_class(parse_params(CONFIGS_DIRECTORY))
class TestSeparateQuants(unittest.TestCase):
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
        from sparseml.pytorch.model_load.helpers import get_session_model
        from sparseml.transformers import oneshot

        try:
            from torch import quantization as torch_quantization
        except Exception:
            torch_quantization = None

        oneshot(
            model=self.model,
            dataset=self.dataset,
            num_calibration_samples=16,
            oneshot_device=self.device,
            recipe=self.first_recipe,
            output_dir=self.output_first,
            clear_sparse_session=False,
        )

        first_model = get_session_model()

        assert not isinstance(
            first_model.model.layers[0].mlp.down_proj, torch_quantization.QuantWrapper
        )
        assert hasattr(first_model.model.embed_tokens, "quantization_scheme")
        session = session_manager.active_session()
        session.reset()

        oneshot(
            model=self.output_first,
            dataset=self.dataset,
            num_calibration_samples=16,
            oneshot_device=self.device,
            recipe=self.second_recipe,
            clear_sparse_session=False,
            output_dir=self.output_second,
        )

        second_model = get_session_model()
        # linear and embeddings should be quantized now
        assert isinstance(
            second_model.model.layers[0].mlp.down_proj, torch_quantization.QuantWrapper
        )
        assert hasattr(second_model.model.embed_tokens, "quantization_scheme")

    def tearDown(self):
        shutil.rmtree(self.output)
