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

import shutil
from copy import copy

import pytest
from transformers import AutoModel

from sparseml.transformers.sparsification.modification import modify_model
from sparseml.transformers.sparsification.modification.registry import (
    ModificationRegistry,
)
from sparsezoo import Model


@pytest.fixture
def model(tmpdir):
    stub = "zoo:mobilebert-squad_wikipedia_bookcorpus-14layer_pruned50.4block_quantized"
    model = Model(stub, tmpdir)
    yield AutoModel.from_pretrained(model.training.path)
    shutil.rmtree(tmpdir)


def test_modify_model_without_actual_modification(model):
    # test to check that the model is not
    # modified if no modification is registered
    # the attribute `training` should not be changed
    is_training = copy(model.training)
    model = modify_model(model)
    assert model.training == is_training == False  # noqa E712


def test_modify_model(model):
    # test to check that the model is modified
    # if a modification is registered
    # the attribute `training` should be changed
    # to True
    @ModificationRegistry.register(name="MobileBertModel")
    def dummy_modification(model):
        model.training = True
        return model

    is_training = copy(model.training)
    model = modify_model(model)
    assert model.training != is_training
