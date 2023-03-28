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


_expected_summaries = {
    "OperationCounts": {
        "ConvReLU2d": 2,
        "MaxPool2d": 1,
        "Flatten": 1,
        "Linear": 1,
    },
    "ParameterCounts": {
        "0.0.module.weight": 144,
        "0.0.module.bias": 16,
        "0.2.module.weight": 4608,
        "0.2.module.bias": 32,
        "1.1.module.weight": 62720,
        "1.1.module.bias": 10,
    },
    "QuantizedOperations/count": 4,
    "QuantizedOperations/percent": 0.8,
    "PrunedParameters/count": 1,
    "PrunedParameters/percent": 0.16666666666666666,
}

_expected_pruning = {
    "0.0.module.weight/count": 0,
    "0.0.module.weight/percent": 0.0,
    "0.0.module.bias/count": 0,
    "0.0.module.bias/percent": 0.0,
    "0.2.module.weight/count": 0,
    "0.2.module.weight/percent": 0.0,
    "0.2.module.bias/count": 0,
    "0.2.module.bias/percent": 0.0,
    "1.1.module.weight/count": 31360,
    "1.1.module.weight/percent": 0.5,
    "1.1.module.bias/count": 0,
    "1.1.module.bias/percent": 0.0,
}

_expected_quantization = {
    "ConvReLU2d/enabled": True,
    "ConvReLU2d/precision/weights/num_bits": 4,
    "ConvReLU2d/precision/input_activations/num_bits": 8,
    "ConvReLU2d_1/enabled": True,
    "ConvReLU2d_1/precision/weights/num_bits": 4,
    "ConvReLU2d_1/precision/input_activations/num_bits": 8,
    "MaxPool2d/enabled": True,
    "MaxPool2d/precision/weights/num_bits": 4,
    "MaxPool2d/precision/input_activations/num_bits": 8,
    "Flatten/enabled": False,
    "Flatten/precision": None,
    "Linear/enabled": True,
    "Linear/precision/weights/num_bits": 4,
    "Linear/precision/input_activations/num_bits": 8,
}


@pytest.mark.parametrize(
    "model, expected_summaries, expected_pruning, expected_quantization",
    [
        (
            _create_test_model(quantization_recipe=QUANT_RECIPE),
            _expected_summaries,
            _expected_pruning,
            _expected_quantization,
        )
    ],
)
class TestSparsificationModels:
    @pytest.fixture()
    def setup(self, model, expected_summaries, expected_pruning, expected_quantization):
        self.expected_summaries = expected_summaries
        self.expected_pruning = expected_pruning
        self.expected_quantization = expected_quantization

        yield model

    def test_sparsification_summaries(self, setup):
        sparsification_summary = SparsificationSummaries.from_module(module=setup)
        for tag, item in sparsification_summary.loggable_items():
            assert (
                self.expected_summaries[tag.replace("SparsificationSummaries/", "")]
                == item
            )

    def test_sparsification_pruning(self, setup):
        sparsification_pruning = SparsificationPruning.from_module(module=setup)
        for tag, item in sparsification_pruning.loggable_items():
            assert (
                self.expected_pruning[
                    tag.replace("SparsificationPruning/SparseParameters/", "")
                ]
                == item
            )

    def test_sparsification_quantization(self, setup):
        sparsification_quantization = SparsificationQuantization.from_module(
            module=setup
        )
        for tag, item in sparsification_quantization.loggable_items():
            assert (
                self.expected_quantization[
                    tag.replace("SparsificationQuantization/", "")
                ]
                == item
            )
