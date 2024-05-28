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


from copy import copy

import pytest

from sparseml.modifiers.quantization_legacy.modification import modify_model
from sparseml.modifiers.quantization_legacy.modification.registry import (
    ModificationRegistry,
)
from sparsezoo.utils.registry import _ALIAS_REGISTRY, _REGISTRY, standardize_lookup_name


@pytest.fixture
def model():
    class DummyModel:
        def __init__(self):
            self.modified = False

    yield DummyModel()


@pytest.fixture
def test_registry():

    key_name = standardize_lookup_name("DummyModel")

    def _cleanup_alias_registry(key_name):
        entry_to_remove = _ALIAS_REGISTRY[ModificationRegistry].get(key_name, None)
        if entry_to_remove:
            del _ALIAS_REGISTRY[ModificationRegistry][key_name]

    def _cleanup_main_registry(key_name):
        entry_to_remove = _REGISTRY[ModificationRegistry].get(key_name, None)
        if entry_to_remove:
            del _REGISTRY[ModificationRegistry][key_name]

    _cleanup_alias_registry(key_name)
    _cleanup_main_registry(key_name)
    yield ModificationRegistry


def test_modify_model_without_actual_modification(model):
    is_modified = copy(model.modified)
    model = modify_model(model)
    assert model.modified == is_modified == False  # noqa E712


def test_modify_model(model, test_registry):
    @test_registry.register(name="DummyModel")
    def dummy_modification(model):
        model.modified = True
        return model

    is_modified = copy(model.modified)
    model = modify_model(model)
    assert model.modified != is_modified


def test_disable_modify_model_through_argument(model, test_registry):
    @test_registry.register(name="DummyModel")
    def dummy_modification(model):
        model.modified = True
        return model

    is_modified = copy(model.modified)
    model = modify_model(model, disable=True)
    assert model.modified == is_modified == False  # noqa E712


def test_disable_modify_model_through_env_var(monkeypatch, model, test_registry):
    @test_registry.register(name="DummyModel")
    def dummy_modification(model):
        model.modified = True
        return model

    is_modified = copy(model.modified)
    monkeypatch.setenv("NM_DISABLE_QUANTIZATION_MODIFICATION", "1")
    model = modify_model(model)
    assert model.modified == is_modified == False  # noqa E712
    monkeypatch.undo()
