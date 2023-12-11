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

import pytest
import torch

from sparseml.core import State
from sparseml.core.event import Event, EventType
from sparseml.core.framework import Framework
from sparseml.modifiers.pruning.constant.pytorch import ConstantPruningModifierPyTorch
from sparseml.modifiers.pruning.utils.pytorch.layer_mask import param_mask_name
from sparseml.pytorch.utils import tensor_sparsity
from tests.sparseml.modifiers.conf import setup_modifier_factory
from tests.sparseml.pytorch.helpers import ConvNet, LinearNet


def _induce_sparsity(model, sparsity=0.5):
    """
    Introduces sparsity to the given model by zeroing out weights
    with a probability of sparsity

    :param model: the model to introduce sparsity to
    :param sparsity: the probability of zeroing out a weight
    :return: the model with sparsity introduced
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name:
                param.data = param.mul_(torch.rand_like(param) > sparsity).float()
    return model


def _make_dense(model):
    """
    Makes a model dense by setting all weights to 1

    :param model: the model to make dense
    :return: the model with all dense params
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name:
                param.data = torch.ones_like(param.data).float()
    return model


def _test_models():

    return [
        _induce_sparsity(LinearNet()),
        _induce_sparsity(ConvNet()),
    ]


def _test_optims():
    return [
        torch.optim.Adam,
        torch.optim.SGD,
    ]


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize("model", _test_models())
@pytest.mark.parametrize("optimizer", _test_optims())
def test_constant_pruning_modifier_e2e(model, optimizer):
    expected_sparsities = {
        name: tensor_sparsity(param.data)
        for name, param in model.named_parameters()
        if "weight" in name
    }

    # init modifier with model

    state = State(framework=Framework.pytorch)
    state.update(
        model=model,
        optimizer=optimizer(model.parameters(), lr=0.1),
        start=0,
    )
    modifier = ConstantPruningModifierPyTorch(
        targets="__ALL_PRUNABLE__",
        start=0,
        end=1,
        update=0.5,
    )
    modifier.initialize(state)

    # check mask is added and has correct sparsity

    for _, parameterized_layer in modifier.parameterized_layers_.items():
        mask_name = param_mask_name()
        mask_tensor = parameterized_layer.layer.get_buffer(mask_name)
        data_tensor = parameterized_layer.param.data
        # check mask and data tensors have 0 in the same places
        assert torch.all(mask_tensor == (data_tensor != 0))

    # mess up model sparsity

    model = _make_dense(model)
    manipulated_sparsities = {
        name: tensor_sparsity(param.data)
        for name, param in model.named_parameters()
        if "weight" in name
    }
    assert manipulated_sparsities != expected_sparsities, "Sparsity manipulation failed"

    # apply modifier

    modifier.on_update(state, event=Event(type_=EventType.OPTIM_PRE_STEP))
    modifier.on_update(state, event=Event(type_=EventType.OPTIM_POST_STEP))
    modifier.on_end(state, None)

    # copy old mask settings as finalize will remove them
    #  this is needed to check if a mask was persistent

    old_mask_settings = modifier._mask_settings.copy()
    modifier.finalize(state)

    # check mask is removed
    for layer_param_name, parameterized_layer in modifier.parameterized_layers_.items():
        mask_name = param_mask_name()

        if not old_mask_settings[layer_param_name].persistent:
            assert not hasattr(parameterized_layer.layer, mask_name)

        # mask name should not be in _mask_settings or
        #  _masked_layer_params
        assert layer_param_name not in modifier._mask_settings
        assert layer_param_name not in modifier._masked_layer_params

    # sparsity should restored by ConstantPruningModifierPyTorch

    actual_sparsities = {
        name: tensor_sparsity(param.data)
        for name, param in model.named_parameters()
        if "weight" in name
    }
    assert actual_sparsities == expected_sparsities, "Sparsity was not constant"


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_constant_pruning_pytorch_is_registered():
    from sparseml.core.factory import ModifierFactory
    from sparseml.core.framework import Framework
    from sparseml.modifiers.pruning.constant.pytorch import (
        ConstantPruningModifierPyTorch,
    )

    kwargs = dict(
        start_epoch=5.0,
        end_epoch=15.0,
        targets="__ALL_PRUNABLE__",
    )
    setup_modifier_factory()
    type_ = ModifierFactory.create(
        type_="ConstantPruningModifier",
        framework=Framework.pytorch,
        allow_experimental=False,
        allow_registered=True,
        **kwargs,
    )

    assert isinstance(
        type_, ConstantPruningModifierPyTorch
    ), "PyTorch ConstantPruningModifier not registered"
