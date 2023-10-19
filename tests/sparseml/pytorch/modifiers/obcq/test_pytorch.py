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

from sparseml.core.framework import Framework
from sparseml.core.model import ModifiableModel
from sparseml.modifiers.obcq.pytorch import SparseGPTModifierPyTorch
from tests.sparseml.modifiers.conf import LifecyleTestingHarness, setup_modifier_factory
from tests.sparseml.pytorch.helpers import LinearNet


@pytest.mark.parametrize(
    "sparsity,targets",
    [
        ([0.5, 0.2], "__ALL__"),  # type mismatch
        ([0.2, 0.1, 0.3], ["seq.fc1", "seq.fc2"]),  # length mismatch
        ([0.3, 0.4], ["re:.*fc1", "re:.*fc2"]),  # regex not supported
    ],
)
def test_invalid_layerwise_recipes_raise_exceptions(sparsity, targets):
    setup_modifier_factory()
    model = LinearNet()

    kwargs = dict(
        sparsity=sparsity,
        block_size=128,
        quantize=False,
        targets=targets,
    )
    modifier = SparseGPTModifierPyTorch(**kwargs)
    testing_harness = LifecyleTestingHarness(model=model)

    # confirm invalid layerwise recipes fail at initialization
    with pytest.raises(ValueError):
        modifier.initialize(testing_harness.get_state())


def test_successful_layerwise_recipe():
    setup_modifier_factory()
    model = LinearNet()

    sparsities = [0.5, 0.2]
    targets = ["seq.fc1", "seq.fc2"]
    kwargs = dict(sparsity=sparsities, block_size=128, quantize=False, targets=targets)
    modifier = SparseGPTModifierPyTorch(**kwargs)
    modifier._validate_layerwise_sparisity()
    modifier.model = ModifiableModel(framework=Framework.pytorch, model=model)
    found_compressible_layers = modifier.compressible_layers()

    # ensure layers names successfully match up with model
    assert len(found_compressible_layers) == len(targets)
