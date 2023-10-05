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

import pytest

from sparseml.core.factory import ModifierFactory
from sparseml.core.framework import Framework
from sparseml.core.recipe.recipe import Recipe
from sparseml.modifiers.pruning.constant.pytorch import ConstantPruningModifierPyTorch


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_modifier_is_registered():
    _setup_modifier_factory()
    kwargs = dict(targets=["re:.*weight"])
    type_ = ModifierFactory.create(
        type_="ConstantPruningModifier",
        framework=Framework.pytorch,
        allow_experimental=False,
        allow_registered=True,
        **kwargs,
    )

    assert isinstance(
        type_, ConstantPruningModifierPyTorch
    ), "PyTorch ConstantPruningModifier not registered"


def _setup_modifier_factory():
    ModifierFactory.refresh()
    assert ModifierFactory._loaded, "ModifierFactory not loaded"
