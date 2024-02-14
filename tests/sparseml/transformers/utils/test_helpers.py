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
from collections import OrderedDict

import pytest
import torch
import transformers

from huggingface_hub import snapshot_download
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
def generative_model_path(tmp_path):
    return snapshot_download("roneneldan/TinyStories-1M", local_dir=tmp_path)


@pytest.fixture()
def model_path(tmp_path):
    return Model(
        "zoo:mobilebert-squad_wikipedia_bookcorpus-14layer_pruned50.4block_quantized",
        tmp_path,
    ).training.path


@pytest.fixture()
def sequence_length():
    return 384


@pytest.fixture()
def dummy_inputs():
    input_ids = torch.zeros((1, 32), dtype=torch.int64)
    attention_mask = torch.ones((1, 32), dtype=torch.int64)

    return OrderedDict(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )


@pytest.mark.parametrize(
    "stub",
    [
        "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned95_obs_quant-none",  # noqa E501
    ],
)
def test_is_transformer_model(tmp_path, stub):
    zoo_model = Model(stub, tmp_path)
    source_path = zoo_model.training.path
    assert is_transformer_model(source_path)


@pytest.mark.parametrize(
    "stub",
    [
        "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned95_obs_quant-none",  # noqa E501
    ],
)
def test_save_zoo_directory(stub, tmp_path_factory):
    path_to_training_outputs = tmp_path_factory.mktemp("outputs")
    save_dir = tmp_path_factory.mktemp("save_dir")

    zoo_model = Model(stub, path_to_training_outputs)
    zoo_model.download()

    save_zoo_directory(
        output_dir=save_dir,
        training_outputs_dir=path_to_training_outputs,
    )
    new_zoo_model = Model(str(save_dir))
    assert new_zoo_model.validate(minimal_validation=True, validate_onnxruntime=False)


@pytest.mark.parametrize(
    "model_path, recipe_found",
    [
        ("roneneldan/TinyStories-1M", False),
        ("mgoin/all-MiniLM-L6-v2-quant-ds", True),
        (
            "zoo:mobilebert-squad_wikipedia_bookcorpus-14layer_pruned50.4block_quantized",  # noqa E501
            True,
        ),
    ],
)
def test_infer_recipe_from_model_path(model_path, recipe_found):
    recipe = infer_recipe_from_model_path(model_path)
    if recipe_found:
        assert isinstance(recipe, str)
        return
    assert recipe is None


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


@pytest.fixture()
def model_path_and_recipe_path(tmp_path):
    model_path = tmp_path / "model.onnx"
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.touch()
    model_path.touch()

    return model_path, recipe_path


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


def test_create_fake_dataloader(generative_model_path, sequence_length):
    expected_input_names = ["input_ids", "attention_mask"]
    sequence_length = 32
    num_samples = 2

    model = transformers.AutoModelForCausalLM.from_pretrained(generative_model_path)
    tokenizer = initialize_tokenizer(
        generative_model_path, sequence_length=sequence_length, task="text-generation"
    )
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
