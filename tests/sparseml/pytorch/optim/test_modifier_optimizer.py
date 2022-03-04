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

from sparseml.pytorch.sparsification import SetWeightDecayModifier
from tests.sparseml.pytorch.helpers import ConvNet, create_optim_adam, create_optim_sgd
from tests.sparseml.pytorch.sparsification.test_modifier import ScheduledModifierTest


from tests.sparseml.pytorch.helpers import (  # noqa isort:skip
    test_epoch,
    test_loss,
    test_steps_per_epoch,
)


OPTIMIZER_MODIFIERS = [
    lambda: SetWeightDecayModifier(
        weight_decay=0.999,
        start_epoch=2.0,
        constant_logging=False,
        param_groups=[0],
    ),
    lambda: SetWeightDecayModifier(
        weight_decay=0.75,
        start_epoch=0.0,
        constant_logging=False,
        param_groups=None,
    ),
]


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize("modifier_lambda", OPTIMIZER_MODIFIERS, scope="function")
@pytest.mark.parametrize("model_lambda", [ConvNet], scope="function")
@pytest.mark.parametrize(
    "optim_lambda", [create_optim_sgd, create_optim_adam], scope="function"
)
class TestSetWeightDecayModifierImpl(ScheduledModifierTest):
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

        # get expected param groups and original values
        param_group_idxs = (
            modifier.param_groups
            if modifier.param_groups
            else list(range(len(optimizer.param_groups)))
        )

        # test pre start epoch
        for epoch in range(int(modifier.start_epoch)):
            assert not modifier.update_ready(epoch, test_steps_per_epoch)
        for idx in param_group_idxs:
            assert optimizer.param_groups[idx]["weight_decay"] != modifier.weight_decay

        # test start epoch + update
        assert modifier.update_ready(modifier.start_epoch, test_steps_per_epoch)
        modifier.scheduled_update(
            model, optimizer, modifier.start_epoch, test_steps_per_epoch
        )
        for idx in param_group_idxs:
            assert optimizer.param_groups[idx]["weight_decay"] == modifier.weight_decay


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_set_weight_decay_modifier_yaml():
    weight_decay = 0.0
    start_epoch = 2.0
    param_groups = [0]
    constant_logging = False
    yaml_str = """
        !SetWeightDecayModifier
            weight_decay: {weight_decay}
            start_epoch: {start_epoch}
            constant_logging: {constant_logging}
            param_groups: {param_groups}
        """.format(
        weight_decay=weight_decay,
        start_epoch=start_epoch,
        constant_logging=constant_logging,
        param_groups=param_groups,
    )
    yaml_modifier = SetWeightDecayModifier.load_obj(
        yaml_str
    )  # type: SetWeightDecayModifier
    serialized_modifier = SetWeightDecayModifier.load_obj(
        str(yaml_modifier)
    )  # type: SetWeightDecayModifier
    obj_modifier = SetWeightDecayModifier(
        weight_decay=weight_decay,
        start_epoch=start_epoch,
        constant_logging=constant_logging,
        param_groups=param_groups,
    )

    assert isinstance(yaml_modifier, SetWeightDecayModifier)
    assert (
        yaml_modifier.weight_decay
        == serialized_modifier.weight_decay
        == obj_modifier.weight_decay
    )
    assert (
        yaml_modifier.start_epoch
        == serialized_modifier.start_epoch
        == obj_modifier.start_epoch
    )
    assert (
        yaml_modifier.constant_logging
        == serialized_modifier.constant_logging
        == obj_modifier.constant_logging
    )
    assert (
        yaml_modifier.param_groups
        == serialized_modifier.param_groups
        == obj_modifier.param_groups
    )
