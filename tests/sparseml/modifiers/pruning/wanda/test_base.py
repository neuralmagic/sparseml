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


from sparseml.core.factory import ModifierFactory
from sparseml.core.framework import Framework
from sparseml.modifiers.pruning.wanda.base import WandaPruningModifier
from tests.sparseml.modifiers.conf import setup_modifier_factory


def test_wanda_registered():
    setup_modifier_factory()

    kwargs = dict(targets="__ALL_PRUNABLE__", init_sparsity=0.0, final_sparsity=0.5)
    quant_obj = ModifierFactory.create(
        type_="WandaPruningModifier",
        framework=Framework.general,
        allow_experimental=False,
        allow_registered=True,
        **kwargs,
    )
    assert isinstance(quant_obj, WandaPruningModifier)
