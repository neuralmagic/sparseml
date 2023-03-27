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

import pytest
import torch

from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils.sparsification_info.configs import (
    SparsificationPruning,
    SparsificationQuantization,
    SparsificationSummaries,
)


QUANT_RECIPE = """
!QuantizationModifier
    start_epoch: 0.0
    scheme:
        input_activations:
            num_bits: 8
            symmetric: False
        weights:
            num_bits: 4
            symmetric: True
        scheme_overrides:
            classifier:
                input_activations:
                    num_bits: 8
                    symmetric: False
                weights: null
            Conv2d:
                input_activations:
                    num_bits: 8
                    symmetric: True
        ignore: ["ReLU", "input"]
        """


def _create_test_model(quantization_recipe: str) -> torch.nn.Module:
    sub_model_1 = torch.nn.Sequential(
        torch.nn.Conv2d(1, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
    )
    sub_model_2 = torch.nn.Sequential(
        torch.nn.Flatten(), torch.nn.Linear(32 * 14 * 14, 10)
    )

    model = torch.nn.ModuleList([sub_model_1, sub_model_2])

    # set some weights to zero to simulate pruning
    named_parameters = dict(model.named_parameters())
    named_parameters["1.1.weight"].data[:5, :] = torch.zeros_like(
        named_parameters["1.1.weight"].data
    )[:5, :]

    manager = ScheduledModifierManager.from_yaml(quantization_recipe)
    manager.apply(model)
    return model


@pytest.mark.parametrize(
    "model", [_create_test_model(quantization_recipe=QUANT_RECIPE)]
)
class TestSparsificationModels:
    @pytest.fixture()
    def setup(self, model):
        yield model

    def test_sparsification_summaries(self, setup):
        sparsification_summary = SparsificationSummaries.from_module(module=setup)
        assert sparsification_summary.operation_counts == {
            "ConvReLU2d": 2,
            "Linear": 1,
            "MaxPool2d": 1,
            "Flatten": 1,
        }
        assert sparsification_summary.parameter_counts == {
            "0.0.module.weight": 144,
            "0.0.module.bias": 16,
            "0.2.module.weight": 4608,
            "0.2.module.bias": 32,
            "1.1.module.weight": 62720,
            "1.1.module.bias": 10,
        }
        assert sparsification_summary.pruned.count == 1
        assert pytest.approx(0.166, sparsification_summary.pruned.percent)
        assert sparsification_summary.quantized.count == 4
        assert sparsification_summary.quantized.percent == 0.8

    def test_sparsification_pruning(self, setup):
        sparsification_pruning = SparsificationPruning.from_module(module=setup)
        for name, _ in setup.named_parameters():
            if name == "1.1.module.weight":
                assert sparsification_pruning.sparse_parameters[name].count == 31360
                assert sparsification_pruning.sparse_parameters[name].percent == 0.5
            else:
                assert sparsification_pruning.sparse_parameters[name].count == 0
                assert sparsification_pruning.sparse_parameters[name].percent == 0.0

    def test_sparsification_quantization(self, setup):
        sparsification_quantization = SparsificationQuantization.from_module(
            module=setup
        )
        is_quantization_enabled = {
            "ConvReLU2d": True,
            "ConvReLU2d_1": True,
            "Linear": True,
            "MaxPool2d": True,
            "Flatten": False,
        }
        assert sparsification_quantization.enabled == is_quantization_enabled
        for operation in is_quantization_enabled.keys():
            if operation == "Flatten":
                assert (
                    sparsification_quantization.quantization_scheme[operation] is None
                )
            else:
                assert (
                    sparsification_quantization.quantization_scheme[
                        operation
                    ].weights.num_bits
                    == 4
                )
                assert (
                    sparsification_quantization.quantization_scheme[
                        operation
                    ].input_activations.num_bits
                    == 8
                )
