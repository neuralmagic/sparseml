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

from sparseml.transformers.sparsification.sparse_model import SparseAutoModelForCausalLM
from tests.testing_utils import requires_torch


@requires_torch
class TestGPTQOneShotWithFullScheme(unittest.TestCase):
    def setUp(self):
        import torch

        self.output = "./oneshot_output"
        self.model = "roneneldan/TinyStories-1M"
        self.dataset = "open_platypus"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.recipe = """
        first_stage:
            quant_modifiers:
                    GPTQModifier:
                        ignore: ["lm_head"]
                        sequential_update: True
                        dampening_frac: 0.001
                        block_size: 128
                        targets: ["Linear"]
                        scheme:
                            input_activations: null
                            output_activations: null
                            weights:
                                num_bits: 8
                                type: "int"
                                symmetric: true
                                strategy: "tensor"
                                group_size: 128
        """

    def test_oneshot_application(self):
        from sparseml.transformers import oneshot

        oneshot(
            model=self.model,
            dataset=self.dataset,
            output_dir=self.output,
            overwrite_output_dir=True,
            recipe=self.recipe,
            oneshot_device=self.device,
            num_calibration_samples=9,
        )

        model_loaded = SparseAutoModelForCausalLM.from_pretrained(self.output)

        # Check that the model is quantized
        assert model_loaded.quantization_config is not None

        # Check a specific layer is quantized
        targetted_linear_layer = model_loaded.transformer.h[0].attn.attention.k_proj
        assert hasattr(targetted_linear_layer, "quantization_scheme")

    def tearDown(self):
        shutil.rmtree(self.output)
