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

from sparseml.pytorch.sparsification.pruning import ConstantPruningModifier
from tests.sparseml.pytorch.helpers import LinearNet
from tests.sparseml.pytorch.sparsification.pruning.helpers import (
    state_dict_save_load_test,
)
from tests.sparseml.pytorch.sparsification.test_modifier import (
    ScheduledModifierTest,
    create_optim_adam,
    create_optim_sgd,
)


from tests.sparseml.pytorch.helpers import (  # noqa isort:skip
    test_epoch,
    test_loss,
    test_steps_per_epoch,
)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "modifier_lambda",
    [
        lambda: ConstantPruningModifier(
            params=["re:.*weight"],
        ),
        lambda: ConstantPruningModifier(
            params=["seq.fc1.weight"],
            start_epoch=10.0,
            end_epoch=25.0,
        ),
    ],
    scope="function",
)
@pytest.mark.parametrize("model_lambda", [LinearNet], scope="function")
@pytest.mark.parametrize(
    "optim_lambda",
    [create_optim_sgd, create_optim_adam],
    scope="function",
)
class TestConstantPruningModifier(ScheduledModifierTest):
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

        # check sparsity is not set before
        if modifier.start_epoch >= 0:
            for epoch in range(int(modifier.start_epoch)):
                assert not modifier.update_ready(epoch, test_steps_per_epoch)

        epoch = int(modifier.start_epoch) if modifier.start_epoch >= 0 else 0.0
        assert modifier.update_ready(epoch, test_steps_per_epoch)
        modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)

        if modifier.end_epoch >= 0:
            epoch = int(modifier.end_epoch)
            assert modifier.update_ready(epoch, test_steps_per_epoch)
            modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)

            for epoch in range(
                int(modifier.end_epoch) + 1, int(modifier.end_epoch) + 6
            ):
                assert not modifier.update_ready(epoch, test_steps_per_epoch)

    def test_state_dict_save_load(
        self,
        modifier_lambda,
        model_lambda,
        optim_lambda,
        test_steps_per_epoch,  # noqa: F811
    ):
        state_dict_save_load_test(
            self,
            modifier_lambda,
            model_lambda,
            optim_lambda,
            test_steps_per_epoch,
            False,
        )


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_constant_pruning_yaml():
    start_epoch = 5.0
    end_epoch = 15.0
    params = ["re:.*weight"]
    yaml_str = f"""
    !ConstantPruningModifier
        start_epoch: {start_epoch}
        end_epoch: {end_epoch}
        params: {params}
    """
    yaml_modifier = ConstantPruningModifier.load_obj(
        yaml_str
    )  # type: ConstantPruningModifier
    serialized_modifier = ConstantPruningModifier.load_obj(
        str(yaml_modifier)
    )  # type: ConstantPruningModifier
    obj_modifier = ConstantPruningModifier(
        start_epoch=start_epoch, end_epoch=end_epoch, params=params
    )

    assert isinstance(yaml_modifier, ConstantPruningModifier)
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
    assert yaml_modifier.params == serialized_modifier.params == obj_modifier.params
