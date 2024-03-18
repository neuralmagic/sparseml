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

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from accelerate import init_empty_weights
from sparseml.transformers.utils.helpers import (
    create_fake_dataloader,
    infer_recipe_from_model_path,
    is_transformer_model,
    resolve_recipe_file,
    save_zoo_directory,
)
from sparseml.transformers.utils.initializers import initialize_tokenizer
from sparsezoo import Model


@pytest.fixture()
def generative_model():
    return "roneneldan/TinyStories-1M"


@pytest.fixture()
def bert_model():
    return "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned95_obs_quant-none"  # noqa E501


@pytest.fixture()
def sequence_length():
    return 320


def test_create_fake_dataloader(generative_model, sequence_length):
    config = AutoConfig.from_pretrained(generative_model)
    tokenizer = initialize_tokenizer(
        generative_model, sequence_length=sequence_length, task="text-generation"
    )
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    expected_input_names = ["input_ids", "attention_mask"]
    num_samples = 2
    data_loader, input_names = create_fake_dataloader(
        model=model,
        tokenizer=tokenizer,
        num_samples=num_samples,
    )

    assert input_names == expected_input_names
    for i, sample in enumerate(data_loader):
        assert sample["input_ids"].shape == torch.Size([1, sequence_length])
        assert sample["attention_mask"].shape == torch.Size([1, sequence_length])
        assert set(sample.keys()) == set(expected_input_names)
    assert i == num_samples - 1


def test_is_transformer_model(tmp_path, bert_model):
    zoo_model = Model(bert_model, tmp_path)
    source_path = zoo_model.training.path
    assert is_transformer_model(source_path)
    shutil.rmtree(tmp_path)


def test_infer_recipe_from_local_model_path(tmp_path):
    model_directory_path = tmp_path
    recipe_path = tmp_path / "recipe.yaml"
    model_path = tmp_path / "model.onnx"
    recipe_path.touch()
    model_path.touch()
    recipe = infer_recipe_from_model_path(model_directory_path)
    assert recipe == recipe_path.as_posix()
    recipe = infer_recipe_from_model_path(model_path)
    assert recipe == recipe_path.as_posix()


@pytest.fixture(autouse=True)
def model_path_and_recipe_path(tmp_path):
    model_path = tmp_path / "model.onnx"
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.touch()
    model_path.touch()

    return model_path, recipe_path


@pytest.mark.parametrize(
    "model_path",
    [
        ("roneneldan/TinyStories-1M"),
        ("mgoin/all-MiniLM-L6-v2-quant-ds"),
        ("zoo:mobilebert-squad_wikipedia_bookcorpus-14layer_pruned50.4block_quantized"),
    ],
)
def test_resolve_recipe_file(model_path, model_path_and_recipe_path):
    recipe = model_path_and_recipe_path[1]
    # looks for recipe: .../.../recipe.yaml in model_path
    assert recipe.as_posix() == resolve_recipe_file(
        requested_recipe=recipe, model_path=model_path
    )


def test_resolve_recipe_file_from_local_path(model_path_and_recipe_path):
    model_path, recipe_path = model_path_and_recipe_path
    assert recipe_path.as_posix() == resolve_recipe_file(
        requested_recipe=recipe_path, model_path=model_path
    )
    assert recipe_path.as_posix() == resolve_recipe_file(
        requested_recipe=recipe_path, model_path=os.path.dirname(model_path)
    )
    new_recipe_path = model_path.parent / "new_recipe.yaml"
    new_recipe_path.touch()
    assert new_recipe_path.as_posix() == resolve_recipe_file(
        requested_recipe=new_recipe_path, model_path=model_path
    )


@pytest.mark.parametrize(
    "model, recipe_found",
    [
        ("roneneldan/TinyStories-1M", False),
        ("mgoin/all-MiniLM-L6-v2-quant-ds", True),
        (
            "zoo:mobilebert-squad_wikipedia_bookcorpus-14layer_pruned50.4block_quantized",  # noqa E501
            True,
        ),
    ],
)
def test_infer_recipe_from_model_path(model, recipe_found):
    recipe = infer_recipe_from_model_path(model)
    if recipe_found:
        assert isinstance(recipe, str)
        return
    assert recipe is None


@pytest.mark.parametrize(
    "stub",
    [
        "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned95_obs_quant-none",  # noqa E501
    ],
)
def test_save_zoo_directory(tmp_path, stub):
    path_to_training_outputs = Model(stub).path
    save_dir = tmp_path

    save_zoo_directory(
        output_dir=save_dir,
        training_outputs_dir=path_to_training_outputs,
    )
    zoo_model = Model(str(save_dir))
    assert zoo_model.validate(minimal_validation=True, validate_onnxruntime=False)
    shutil.rmtree(path_to_training_outputs)
    shutil.rmtree(save_dir)
