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

import sparseml
from compressed_tensors import SPARSITY_CONFIG_NAME
from compressed_tensors.config import BitmaskConfig, DenseSparsityConfig
from sparseml.transformers import SparseAutoModelForCausalLM, oneshot
from sparseml.transformers.compression.sparsity_config import SparsityConfigMetadata


@pytest.mark.parametrize(
    "compressed,config,dtype",
    [
        [True, None, torch.float32],
        [False, DenseSparsityConfig(), torch.float16],
        [True, BitmaskConfig(), torch.bfloat16],
        [False, BitmaskConfig(), torch.float32],
        [False, None, torch.float16],
    ],
)
def test_sparse_model_reload(compressed, config, dtype, tmp_path):
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
        precision=dtype,
    )

    model = SparseAutoModelForCausalLM.from_pretrained(
        tmp_path / "oneshot_out", torch_dtype=dtype
    )

    inferred_global_sparsity = SparsityConfigMetadata.infer_global_sparsity(model)
    assert math.isclose(inferred_global_sparsity, 19.6562, rel_tol=1e-3)
    inferred_structure = SparsityConfigMetadata.infer_sparsity_structure()
    assert inferred_structure == "0:0"

    model.save_pretrained(
        tmp_path / "compress_out",
        sparsity_config=config,
        save_compressed=compressed,
    )

    config = AutoConfig.from_pretrained(tmp_path / "compress_out")
    sparsity_config = getattr(config, SPARSITY_CONFIG_NAME, None)
    assert (
        sparsity_config["format"] == "dense"
        if (not compressed and config is None)
        else "sparse_bitmask"
    )
    assert sparsity_config["global_sparsity"] == inferred_global_sparsity
    assert sparsity_config["sparsity_structure"] == inferred_structure

    dense_model = SparseAutoModelForCausalLM.from_pretrained(
        tmp_path / "compress_out", torch_dtype="auto"
    )

    og_state_dict = model.state_dict()
    reconstructed_state_dict = dense_model.state_dict()
    assert len(og_state_dict) == len(reconstructed_state_dict)
    for key in og_state_dict.keys():
        dense_tensor = og_state_dict[key]
        reconstructed_tensor = reconstructed_state_dict[key]
        assert dense_tensor.dtype == reconstructed_tensor.dtype == dtype
        assert torch.equal(dense_tensor, reconstructed_tensor)

    shutil.rmtree(tmp_path)


@pytest.mark.parametrize(
    "skip_compression_stats,save_compressed",
    [[True, True], [True, False], [False, True], [False, False]],
)
def test_dense_model_save(tmp_path, skip_compression_stats, save_compressed):
    sparseml.reset_session()

    model_path = "Xenova/llama2.c-stories15M"
    model = SparseAutoModelForCausalLM.from_pretrained(model_path)

    inferred_global_sparsity = SparsityConfigMetadata.infer_global_sparsity(model)
    assert math.isclose(inferred_global_sparsity, 0.0, rel_tol=1e-3)
    inferred_structure = SparsityConfigMetadata.infer_sparsity_structure()
    assert inferred_structure == "unstructured"

    model.save_pretrained(
        tmp_path / "dense_out",
        skip_compression_stats=skip_compression_stats,
        save_compressed=save_compressed,
    )

    # for models with 0% sparsity no sparsity config is saved regardless
    config = AutoConfig.from_pretrained(tmp_path / "dense_out")
    sparsity_config = getattr(config, SPARSITY_CONFIG_NAME, None)
    assert sparsity_config is None

    shutil.rmtree(tmp_path)
