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
import shutil

import pytest
import torch
from transformers import AutoConfig

import sparseml.core.session as session_manager
from sparseml.transformers import SparseAutoModelForCausalLM, oneshot
from sparseml.transformers.compression import BitmaskConfig, CompressionConfig
from sparseml.transformers.utils.sparse_model import SPARSITY_CONFIG_NAME


@pytest.mark.parametrize(
    "dense,config",
    [[True, None], [False, None], [True, BitmaskConfig()], [False, BitmaskConfig()]],
)
def test_sparse_model_reload(dense, config, tmp_path):
    recipe_str = "tests/sparseml/transformers/obcq/test_tiny2.yaml"
    model_path = "Xenova/llama2.c-stories15M"
    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"
    dataset = "open_platypus"
    concatenate_data = False
    num_calibration_samples = 64
    output_dir = tmp_path / "oneshot_out"
    splits = {"calibration": "train[:10%]"}

    # create a sparse model
    oneshot(
        model=model_path,
        dataset=dataset,
        output_dir=output_dir,
        num_calibration_samples=num_calibration_samples,
        recipe=recipe_str,
        concatenate_data=concatenate_data,
        splits=splits,
        oneshot_device=device,
    )

    model = SparseAutoModelForCausalLM.from_pretrained(tmp_path / "oneshot_out")

    inferred_global_sparsity = CompressionConfig.infer_global_sparsity(model)
    assert math.isclose(inferred_global_sparsity, 19.6562, rel_tol=1e-3)
    inferred_structure = CompressionConfig.infer_sparsity_structure()
    assert inferred_structure == "0:0"

    SparseAutoModelForCausalLM.save_pretrained(
        model, tmp_path / "compress_out", sparsity_config=config, save_dense=dense
    )
    config = AutoConfig.from_pretrained(tmp_path / "compress_out")
    sparsity_config = getattr(config, SPARSITY_CONFIG_NAME, None)
    assert (
        sparsity_config["format"] == "dense"
        if (dense and config is None)
        else "sparse_bitmask"
    )
    assert sparsity_config["global_sparsity"] == inferred_global_sparsity
    assert sparsity_config["sparsity_structure"] == inferred_structure

    dense_model = SparseAutoModelForCausalLM.from_pretrained(tmp_path / "compress_out")

    og_state_dict = model.state_dict()
    reconstructed_state_dict = dense_model.state_dict()
    assert len(og_state_dict) == len(reconstructed_state_dict)
    for key in og_state_dict.keys():
        dense_tensor = og_state_dict[key]
        reconstructed_tensor = reconstructed_state_dict[key]
        assert torch.equal(dense_tensor.cpu(), reconstructed_tensor.cpu())

    shutil.rmtree(tmp_path)


def test_dense_model_save(tmp_path):
    session_manager.active_session().reset()

    model_path = "Xenova/llama2.c-stories15M"
    model = SparseAutoModelForCausalLM.from_pretrained(model_path)

    inferred_global_sparsity = CompressionConfig.infer_global_sparsity(model)
    assert math.isclose(inferred_global_sparsity, 0.0, rel_tol=1e-3)
    inferred_structure = CompressionConfig.infer_sparsity_structure()
    assert inferred_structure == "unstructured"

    SparseAutoModelForCausalLM.save_pretrained(model, tmp_path / "dense_out")
    config = AutoConfig.from_pretrained(tmp_path / "dense_out")
    sparsity_config = getattr(config, SPARSITY_CONFIG_NAME, None)
    assert sparsity_config is None

    shutil.rmtree(tmp_path)
