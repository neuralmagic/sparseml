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

from sparseml.transformers.sparsification.modification import modify_model
from sparseml.transformers.sparsification.modification.registry import (
    ModificationRegistry,
)


@pytest.fixture
def model():
    class DummyModel:
        def __init__(self):
            self.modified = False

    yield DummyModel()


def test_modify_model_without_actual_modification(model):
    is_modified = copy(model.modified)
    model = modify_model(model)
    assert model.modified == is_modified == False  # noqa E712


def test_modify_model(model):
    @ModificationRegistry.register(name="DummyModel")
    def dummy_modification(model):
        model.modified = True
        return model

    is_modified = copy(model.modified)
    model = modify_model(model)
    assert model.modified != is_modified
