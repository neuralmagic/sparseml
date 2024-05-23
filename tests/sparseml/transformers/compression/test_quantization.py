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
import tempfile
import unittest

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator

from compressed_tensors.quantization import fake_quantize
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
from tests.testing_utils import parse_params, requires_gpu, requires_torch


CONFIGS_DIRECTORY = "tests/sparseml/transformers/compression/configs"


@requires_torch
@requires_gpu
@pytest.mark.integration
@parameterized_class(parse_params(CONFIGS_DIRECTORY))
class TestQuantizationMatches(unittest.TestCase):
    """
    Tests new compressed-tensors quantization format matches performance with the old
    sparseml format. For setup, this class runs a full oneshot run with both an old and
    new quantization recipe that should be equivalent. Then tests the following:
        - quantization structure matches after oneshot
        - quantized weights match
        - decompressing the new model has the expected weights on reload
        - no perplexity regression from the old quantization framework, asserts we are
            no more than 2% on perplexity
    """

    old_recipe = None
    new_recipe = None
    model_stub = None
    dataset = "open_platypus"
    old_output = "tiny_llama_old"
    new_output = "tiny_llama_new"
    max_seq_length = 512
    num_comparisons = 64

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

    @staticmethod
    def _run_oneshot(model, recipe, dataset, output_dir):
        num_calibration_samples = 256
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
            clear_sparse_session=True,
        )

    def _get_quant_info_old(self, model):
        quant_info_weights = {}
        quant_info_inputs = {}
        for name, module in model.named_modules():
            if hasattr(module, "weight_fake_quant"):
                scale = module.weight_fake_quant.scale
                zp = module.weight_fake_quant.zero_point
                weight = module.weight_fake_quant(module.weight)
                quant_info_weights[name] = (scale, zp, weight)
            elif hasattr(module, "quant"):
                scale = module.quant.activation_post_process.scale
                zp = module.quant.activation_post_process.zero_point
                quant_info_inputs[name] = (scale, zp)

        return quant_info_weights, quant_info_inputs

    def _get_quant_info_new(self, model):
        quant_info_weights = {}
        quant_info_inputs = {}
        for name, module in model.named_modules():
            if is_module_quantized(module):
                if module.quantization_scheme.weights is not None:
                    quant_info_weights[name] = (
                        module.weight_scale,
                        module.weight_zero_point,
                        fake_quantize(
                            module.weight,
                            module.weight_scale,
                            module.weight_zero_point,
                            module.quantization_scheme.weights,
                        ),
                    )
                if module.quantization_scheme.input_activations is not None:
                    quant_info_inputs[name] = (
                        module.input_scale,
                        module.input_zero_point,
                    )

        return quant_info_weights, quant_info_inputs

    def _get_dataloader(self, data_args, tokenizer):
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

    def test_quantization_counts(self):
        old_quant_weights, old_quant_inputs = self._get_quant_info_old(self.model_old)
        new_quant_weights, new_quant_inputs = self._get_quant_info_new(self.model_new)

        assert len(old_quant_weights) == len(new_quant_weights)
        assert len(old_quant_inputs) == len(new_quant_inputs)

    def test_quantization_matches(self):
        old_quant_weights, _ = self._get_quant_info_old(self.model_old)
        new_quant_weights, _ = self._get_quant_info_new(self.model_new)

        for name, (o_scale, o_zp, _) in old_quant_weights.items():
            if name.endswith(".module"):
                name = name[:-7]
            n_scale, n_zp, _ = new_quant_weights[name]
            if n_scale.ndim == 2:  # channelwise
                n_scale = n_scale[:, 0]
                n_zp = n_zp[:, 0]
            elif n_scale.ndim == 0:  # tensor
                n_scale = torch.unsqueeze(n_scale, 0)
                n_zp = torch.unsqueeze(n_zp, 0)

            assert torch.all(
                torch.isclose(o_scale.cpu(), n_scale.cpu(), atol=1e-3, rtol=1e-3)
            )

    def test_quantization_reload(self):
        model_reloaded = SparseAutoModelForCausalLM.from_pretrained(
            os.path.join(self.test_dir, self.new_output)
        )

        og_weights, og_inputs = self._get_quant_info_new(self.model_new)
        reloaded_weights, reloaded_inputs = self._get_quant_info_new(model_reloaded)

        for name, (o_scale, o_zp, _) in og_weights.items():
            n_scale, n_zp, _ = reloaded_weights[name]
            assert torch.equal(o_scale.cpu(), n_scale.cpu())
            assert torch.equal(o_zp.cpu(), n_zp.cpu())

        for name, (o_scale, o_zp) in og_inputs.items():
            n_scale, n_zp = reloaded_inputs[name]
            assert torch.equal(o_scale.cpu(), n_scale.cpu())
            assert torch.equal(o_zp.cpu(), n_zp.cpu())

    @torch.no_grad()
    def test_perplexity(self):
        tokenizer = SparseAutoTokenizer.from_pretrained(self.model_stub)
        data_args = DataTrainingArguments(
            dataset="wikitext",
            dataset_config_name="wikitext-2-raw-v1",
            max_seq_length=self.max_seq_length,
            concatenate_data=True,
        )
        dataloader = self._get_dataloader(data_args, tokenizer)

        total_ppl_old = 0.0
        total_ppl_new = 0.0
        total_non_nan = 0
        for idx, sample in enumerate(dataloader):
            if idx >= self.num_comparisons:
                break
            output_new = self.model_new(**tensors_to_device(sample, "cuda:0"))
            output_old = self.model_old(**tensors_to_device(sample, "cuda:0"))
            if torch.isnan(output_old.loss) and torch.isnan(output_new.loss):
                continue
            total_ppl_old += torch.exp(output_old.loss).item()
            total_ppl_new += torch.exp(output_new.loss).item()
            total_non_nan += 1

        avg_ppl_ratio = (total_ppl_new / total_non_nan) / (
            total_ppl_old / total_non_nan
        )
        assert avg_ppl_ratio <= 1.02

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)
        del cls.model_new
        del cls.model_old
        torch.cuda.empty_cache()
