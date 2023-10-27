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

from sparseml.core.factory import ModifierFactory
from sparseml.core.framework import Framework
from sparseml.modifiers.pruning.wanda.pytorch import (
    WandaPruningModifierPyTorch,
    wanda_score,
)
from tests.sparseml.modifiers.conf import setup_modifier_factory


def test_wanda_registered():
    setup_modifier_factory()

    kwargs = dict(targets="__ALL_PRUNABLE__", init_sparsity=0.0, final_sparsity=0.5)
    quant_obj = ModifierFactory.create(
        type_="WandaPruningModifier",
        framework=Framework.pytorch,
        allow_experimental=False,
        allow_registered=True,
        **kwargs,
    )
    assert isinstance(quant_obj, WandaPruningModifierPyTorch)


def _get_weight_l2_norm_and_expected_score():
    weight = torch.tensor(
        [
            [4, 0, 1, -1],
            [3, -2, -1, -3],
            [-3, 1, 0, 2],
        ]
    )

    l2_norm = torch.tensor([1, 2, 8, 3])

    expected = torch.tensor(
        [
            [4, 0, 8, 3],
            [3, 4, 8, 9],
            [3, 2, 0, 6],
        ]
    )

    return weight, l2_norm, expected


@pytest.mark.parametrize(
    "weight, l2_norm, expected_scores", [_get_weight_l2_norm_and_expected_score()]
)
def test_wanda_scores(weight, l2_norm, expected_scores):
    actual_scores = wanda_score(weight, l2_norm)
    assert torch.equal(actual_scores, expected_scores), "wanda scores are not correct"
