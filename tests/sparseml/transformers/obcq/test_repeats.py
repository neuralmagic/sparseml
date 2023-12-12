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

import torch

import sparseml.core.session as session_manager
from sparseml.pytorch.utils.helpers import tensor_sparsity
from sparseml.transformers.sparsification.obcq.obcq import one_shot
from sparseml.utils.pytorch import qat_active


def test_consecutive_runs(tmp_path):
    tiny_model_path = "Xenova/llama2.c-stories15M"
    first_recipe = "tests/sparseml/transformers/obcq/test_tiny.yaml"
    second_recipe = "tests/sparseml/transformers/obcq/test_additional_sparsity.yaml"
    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"

    # test recipe with 50% sparsity, quantization and smoothquant
    first_tiny_model = one_shot(
        model_path=tiny_model_path,
        dataset_name="open_platypus",
        num_samples=16,
        device=device,
        recipe_file=first_recipe,
        deploy_dir=tmp_path,
        do_save=True,
    )
    layer_0_sparse = tensor_sparsity(
        first_tiny_model.model.layers[0].self_attn.k_proj.module.weight
    )
    assert math.isclose(layer_0_sparse.item(), 0.5, rel_tol=1e-3)
    assert qat_active(first_tiny_model)

    session = session_manager.active_session()
    session_recipe = session.lifecycle.recipe_container.compiled_recipe
    stages = [stage.group for stage in session_recipe.stages]
    assert len(stages) == 1
    assert "test" in stages
    session.reset()

    # reload saved model and up sparsity to 0.7
    second_tiny_model = one_shot(
        model_path=tmp_path / "obcq_deployment",
        dataset_name="open_platypus",
        num_samples=16,
        device=device,
        recipe_file=second_recipe,
    )
    layer_0_sparse = tensor_sparsity(
        second_tiny_model.model.layers[0].self_attn.k_proj.module.weight
    )
    assert math.isclose(layer_0_sparse.item(), 0.7, rel_tol=1e-3)
    assert qat_active(second_tiny_model)

    session = session_manager.active_session()
    session_recipe = session.lifecycle.recipe_container.compiled_recipe
    stages = [stage.group for stage in session_recipe.stages]
    assert len(stages) == 2
    assert "test" in stages
    assert "test_second" in stages
