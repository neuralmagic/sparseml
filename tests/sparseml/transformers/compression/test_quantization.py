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

import math
import os
import shutil
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator

from compressed_tensors.quantization.utils import is_module_quantized
from parameterized import parameterized_class
from sparseml.pytorch.utils import tensors_to_device
from sparseml.transformers import (
    SparseAutoModelForCausalLM,
    SparseAutoTokenizer,
    oneshot,
)
from sparseml.transformers.finetune.data import TextGenerationDataset
from sparseml.transformers.finetune.data.data_args import DataTrainingArguments
from tests.testing_utils import requires_gpu, requires_torch


@requires_torch
@requires_gpu
@parameterized_class(
    ("old_recipe", "new_recipe"),
    [
        (
            "tests/sparseml/transformers/compression/recipes/old_quant_full.yaml",
            "tests/sparseml/transformers/compression/recipes/new_quant_full.yaml",
        ),
        (
            "tests/sparseml/transformers/compression/recipes/old_quant_weight.yaml",
            "tests/sparseml/transformers/compression/recipes/new_quant_weight.yaml",
        ),
    ],
)
class TestQuantizationMatches(unittest.TestCase):
    old_recipe = None
    new_recipe = None
    model_stub = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    dataset = "open_platypus"
    old_output = "tiny_llama_old"
    new_output = "tiny_llama_new"
    max_seq_length = 512
    num_comparisons = 2

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()

        cls.model_old = SparseAutoModelForCausalLM.from_pretrained(
            cls.model_stub, device_map="cuda:0"
        )
        cls._run_oneshot(
            cls.model_old,
            cls.old_recipe,
            cls.dataset,
            os.path.join(cls.test_dir, cls.old_output),
        )

        cls.model_new = SparseAutoModelForCausalLM.from_pretrained(
            cls.model_stub, device_map="cuda:0"
        )
        cls._run_oneshot(
            cls.model_new,
            cls.new_recipe,
            cls.dataset,
            os.path.join(cls.test_dir, cls.new_output),
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    @staticmethod
    def _run_oneshot(model, recipe, dataset, output_dir):
        num_calibration_samples = 512
        max_seq_length = 512
        pad_to_max_length = False

        oneshot(
            model=model,
            dataset=dataset,
            overwrite_output_dir=True,
            output_dir=output_dir,
            max_seq_length=max_seq_length,
            num_calibration_samples=num_calibration_samples,
            recipe=recipe,
            pad_to_max_length=pad_to_max_length,
        )

    def _get_quant_info_old(self, model):
        quant_info_weights = {}
        quant_info_inputs = {}
        for name, module in model.named_modules():
            if hasattr(module, "weight_fake_quant"):
                scale = module.weight_fake_quant.scale.item()
                zp = module.weight_fake_quant.zero_point.item()
                quant_info_weights[name] = (scale, zp)
            elif hasattr(module, "quant"):
                scale = module.quant.activation_post_process.scale.item()
                zp = module.quant.activation_post_process.zero_point.item()
                quant_info_inputs[name] = (scale, zp)

        return quant_info_weights, quant_info_inputs

    def _get_quant_info_new(self, model):
        quant_info_weights = {}
        quant_info_inputs = {}
        for name, module in model.named_modules():
            if is_module_quantized(module):
                if module.quantization_scheme.weights is not None:
                    quant_info_weights[name] = (
                        module.weight_scale.item(),
                        module.weight_zero_point.item(),
                    )
                if module.quantization_scheme.input_activations is not None:
                    quant_info_inputs[name] = (
                        module.input_scale.item(),
                        module.input_zero_point.item(),
                    )

        return quant_info_weights, quant_info_inputs

    def test_quantization_counts(self):
        old_quant_weights, old_quant_inputs = self._get_quant_info_old(self.model_old)
        new_quant_weights, new_quant_inputs = self._get_quant_info_new(self.model_new)

        assert len(old_quant_weights) == len(new_quant_weights)
        assert len(old_quant_inputs) == len(new_quant_inputs)

    def test_quantization_scale_and_zp(self):
        old_quant_weights, old_quant_inputs = self._get_quant_info_old(self.model_old)
        new_quant_weights, new_quant_inputs = self._get_quant_info_new(self.model_new)

        for name, (o_scale, o_zp) in old_quant_weights.items():
            if name.endswith(".module"):
                name = name[:-7]
            n_scale, n_zp = new_quant_weights[name]
            assert math.isclose(o_scale, n_scale, abs_tol=1e-3, rel_tol=1e-3)
            assert o_zp == n_zp

        for name, (o_scale, o_zp) in old_quant_inputs.items():
            n_scale, n_zp = new_quant_inputs[name]
            assert math.isclose(o_scale, n_scale, abs_tol=1e-3, rel_tol=1e-3)
            assert o_zp == n_zp

    def test_quantization_reload(self):
        model_reloaded = SparseAutoModelForCausalLM.from_pretrained(
            self.test_dir / self.new_output
        )

        og_weights, og_inputs = self._get_quant_info_new(self.model_new)
        reloaded_weights, reloaded_inputs = self._get_quant_info_new(model_reloaded)

        for name, (o_scale, o_zp) in og_weights.items():
            n_scale, n_zp = reloaded_weights[name]
            assert o_scale == n_scale
            assert o_zp == n_zp

        for name, (o_scale, o_zp) in og_inputs.items():
            n_scale, n_zp = reloaded_inputs[name]
            assert o_scale == n_scale
            assert o_zp == n_zp

    def _get_dataloader(self, dataset_name, tokenizer):
        data_args = DataTrainingArguments(
            dataset=dataset_name,
            max_seq_length=self.max_seq_length,
            pad_to_max_length=False,
        )
        dataset_manager = TextGenerationDataset.load_from_registry(
            data_args.dataset,
            data_args=data_args,
            split="train",
            tokenizer=tokenizer,
        )
        calib_dataset = dataset_manager.tokenize_and_process(
            dataset_manager.get_raw_dataset()
        )
        data_loader = DataLoader(
            calib_dataset,
            batch_size=1,
            collate_fn=DefaultDataCollator(),
            sampler=torch.utils.data.RandomSampler(calib_dataset),
        )

        return data_loader

    def test_perplexity(self):
        tokenizer = SparseAutoTokenizer.from_pretrained(self.model_stub)
        dataloader = self._get_dataloader(self.dataset, tokenizer)

        for idx, sample in enumerate(dataloader):
            if idx >= self.num_comparisons:
                break
            sample_new = tensors_to_device(sample, "cuda:0")
            sample_old = tensors_to_device(sample, "cuda:0")
            output_new = self.model_new(**sample_new)
            output_old = self.model_old(**sample_old)
            ppl_ratio = (
                torch.exp(output_new.loss).item() / torch.exp(output_old.loss).item()
            )
            assert abs(1.0 - ppl_ratio) < 0.05
