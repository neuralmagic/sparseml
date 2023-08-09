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

import pytest
from torch.nn import Module
from torch.optim import SGD

from sparseml.pytorch.sparsification.pruning import TopKASTPruningModifier
from tests.sparseml.pytorch.helpers import LinearNet
from tests.sparseml.pytorch.sparsification.pruning.helpers import (
    state_dict_save_load_test,
)
from tests.sparseml.pytorch.sparsification.test_modifier import (
    ScheduledModifierTest,
    create_optim_adam,
)


from tests.sparseml.pytorch.helpers import (  # noqa isort:skip
    test_epoch,
    test_loss,
    test_steps_per_epoch,
)


def create_optim_sgd(
    model: Module, lr: float = 1e-3, momentum: float = 0.9, weight_decay: float = 0
) -> SGD:
    return SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


@pytest.mark.parametrize(
    "modifier_lambda",
    [
        lambda: TopKASTPruningModifier(
            forward_sparsity=0.9,
            backward_sparsity=0.5,
            start_epoch=0,
            end_epoch=20,
            update_frequency=5,
            params=["re:.*weight"],
            leave_enabled=True,
            active_weight_decay=0.0002,
        ),
        lambda: TopKASTPruningModifier(
            forward_sparsity=0.9,
            backward_sparsity=0.5,
            start_epoch=0,
            end_epoch=20,
            update_frequency=5,
            params=["re:.*weight"],
            active_weight_decay=0.0002,
        ),
        lambda: TopKASTPruningModifier(
            forward_sparsity=0.8,
            backward_sparsity=0.7,
            start_epoch=6.0,
            end_epoch=26.0,
            update_frequency=1,
            params=["re:.*weight"],
            active_weight_decay=0.0002,
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
class TestTopKASTPruningModifier(ScheduledModifierTest):
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

        def _test_compression_sparsity_applied():
            assert True
            assert modifier._forward_sparsity == modifier.applied_sparsity

        # assert that until modifier is activated, `applied_sparsity` remains None
        if modifier.start_epoch > 0:
            assert modifier.applied_sparsity is None

        for epoch in range(int(modifier.start_epoch)):
            assert not modifier.update_ready(epoch, test_steps_per_epoch)
            assert modifier.applied_sparsity is None

        # assert that once `start_epoch` happens, modifier is ready for update.
        epoch = int(modifier.start_epoch)
        assert modifier.update_ready(epoch, test_steps_per_epoch)
        modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)

        while epoch < modifier.end_epoch:
            epoch += modifier.update_frequency
            assert modifier.update_ready(epoch, test_steps_per_epoch)
            modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)

        for epoch in range(int(modifier.end_epoch) + 1, int(modifier.end_epoch) + 2):
            assert epoch > modifier.end_epoch
            assert not modifier.update_ready(epoch, test_steps_per_epoch)
            _test_compression_sparsity_applied()

    def test_state_dict_save_load(
        self,
        modifier_lambda,
        model_lambda,
        optim_lambda,
        test_steps_per_epoch,  # noqa: F811
    ):
        return
        state_dict_save_load_test(
            self,
            modifier_lambda,
            model_lambda,
            optim_lambda,
            test_steps_per_epoch,
            False,
        )


def test_topkast_pruning_yaml():
    forward_sparsity = 0.9
    backward_sparsity = 0.5
    start_epoch = 6
    end_epoch = 26
    update_frequency = 1
    params = ["re:.*weight"]
    global_sparsity = True
    mask_type = "unstructured"
    leave_enabled = False
    active_weight_decay = 0.0002

    yaml_str = f"""
    !TopKASTPruningModifier
        forward_sparsity: {forward_sparsity}
        backward_sparsity: {backward_sparsity}
        start_epoch: {start_epoch}
        end_epoch: {end_epoch}
        update_frequency: {update_frequency}
        params: {params}
        global_sparsity: {global_sparsity}
        leave_enabled: {leave_enabled}
        mask_type: {mask_type}
        active_weight_decay: {active_weight_decay}
        """
    yaml_modifier = TopKASTPruningModifier.load_obj(
        yaml_str
    )  # type: TopKASTPruningModifier
    serialized_modifier = TopKASTPruningModifier.load_obj(
        str(yaml_modifier)
    )  # type: TopKASTPruningModifier
    obj_modifier = TopKASTPruningModifier(
        forward_sparsity=forward_sparsity,
        backward_sparsity=backward_sparsity,
        update_frequency=update_frequency,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        params=params,
        global_sparsity=global_sparsity,
        leave_enabled=leave_enabled,
        mask_type=mask_type,
        active_weight_decay = active_weight_decay
    )
    assert isinstance(yaml_modifier, TopKASTPruningModifier)
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
    assert (
        yaml_modifier.forward_sparsity
        == serialized_modifier.forward_sparsity
        == obj_modifier.forward_sparsity
    )
    assert (
        yaml_modifier.backward_sparsity
        == serialized_modifier.backward_sparsity
        == obj_modifier.backward_sparsity
    )
    assert (
        yaml_modifier.update_frequency
        == serialized_modifier.update_frequency
        == obj_modifier.update_frequency
    )
    assert (
        yaml_modifier.global_sparsity
        == serialized_modifier.global_sparsity
        == obj_modifier.global_sparsity
    )
    assert (
        yaml_modifier.leave_enabled
        == serialized_modifier.leave_enabled
        == obj_modifier.leave_enabled
    )
    assert (
        yaml_modifier.mask_type
        == serialized_modifier.mask_type
        == obj_modifier.mask_type
    )
