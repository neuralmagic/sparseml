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

from sparseml.pytorch.nn import Identity
from sparseml.pytorch.sparsification import LayerPruningModifier
from sparseml.pytorch.utils import get_layer
from tests.sparseml.pytorch.helpers import FlatMLPNet
from tests.sparseml.pytorch.optim.test_modifier import (
    ScheduledUpdateModifierTest,
    create_optim_adam,
    create_optim_sgd,
)


from tests.sparseml.pytorch.helpers import (  # noqa isort:skip
    test_epoch,
    test_loss,
    test_steps_per_epoch,
)


@pytest.mark.parametrize(
    "modifier_lambda",
    [
        lambda: LayerPruningModifier(
            layers=["seq.fc2", "seq.act2"],
            start_epoch=0.0,
            end_epoch=15.0,
            update_frequency=1.0,
        ),
        lambda: LayerPruningModifier(
            layers=["seq.fc2", "seq.act2"],
            start_epoch=10.0,
            end_epoch=25.0,
            update_frequency=1.0,
        ),
        lambda: LayerPruningModifier(
            layers="__ALL_PRUNABLE__",
            start_epoch=10.0,
            end_epoch=25.0,
        ),
    ],
    scope="function",
)
@pytest.mark.parametrize("model_lambda", [FlatMLPNet], scope="function")
@pytest.mark.parametrize(
    "optim_lambda",
    [create_optim_sgd, create_optim_adam],
    scope="function",
)
class TestLayerPruningModifier(ScheduledUpdateModifierTest):
    def test_lifecycle(
        self,
        modifier_lambda,
        model_lambda,
        optim_lambda,
        test_steps_per_epoch,  # noqa: F811
    ):
        modifier = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)
        self.initialize_helper(modifier, model)
        assert len(modifier._layer_modules) > 0
        if modifier.start_epoch > 0:
            for (name, mod) in modifier._layer_modules.items():
                assert mod is None
                assert not isinstance(get_layer(name, model), Identity)

            # check sparsity is not set before
            for epoch in range(int(modifier.start_epoch)):
                assert not modifier.update_ready(epoch, test_steps_per_epoch)

            for (name, mod) in modifier._layer_modules.items():
                assert mod is None
                assert not isinstance(get_layer(name, model), Identity)

            epoch = int(modifier.start_epoch)
            assert modifier.update_ready(epoch, test_steps_per_epoch)
            modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)
        else:
            epoch = 0

        for (name, mod) in modifier._layer_modules.items():
            assert mod is not None
            assert isinstance(get_layer(name, model), Identity)

        # check forward pass
        input_shape = model_lambda.layer_descs()[0].input_size
        test_batch = torch.randn(10, *input_shape)
        _ = model(test_batch)

        end_epoch = (
            modifier.end_epoch if modifier.end_epoch > -1 else modifier.start_epoch + 10
        )

        while epoch < end_epoch - 0.1:
            epoch += 0.1
            assert modifier.update_ready(epoch, test_steps_per_epoch)
            modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)

        _ = model(test_batch)  # check forward pass

        if modifier.end_epoch > -1:
            epoch = int(modifier.end_epoch)
            assert modifier.update_ready(epoch, test_steps_per_epoch)
            modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)

            for (name, mod) in modifier._layer_modules.items():
                assert mod is None
                assert not isinstance(get_layer(name, model), Identity)


def _test_pruning_modifier_serialization_vals(
    yaml_modifier, serialized_modifier, obj_modifier
):
    assert (
        yaml_modifier.init_sparsity
        == serialized_modifier.init_sparsity
        == obj_modifier.init_sparsity
    )
    assert (
        yaml_modifier.final_sparsity
        == serialized_modifier.final_sparsity
        == obj_modifier.final_sparsity
    )
    assert (
        yaml_modifier.start_epoch
        == serialized_modifier.start_epoch
        == obj_modifier.start_epoch
    )
    assert (
        yaml_modifier.end_epoch
        == serialized_modifier.end_epoch
        == obj_modifier.end_epoch
    )
    assert (
        yaml_modifier.update_frequency
        == serialized_modifier.update_frequency
        == obj_modifier.update_frequency
    )
    assert yaml_modifier.params == serialized_modifier.params == obj_modifier.params
    assert (
        yaml_modifier.inter_func
        == serialized_modifier.inter_func
        == obj_modifier.inter_func
    )
    assert (
        str(yaml_modifier.mask_type)
        == str(serialized_modifier.mask_type)
        == str(obj_modifier.mask_type)
    )
