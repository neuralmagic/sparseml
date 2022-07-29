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
from typing import Any, Dict, Optional

import pytest
import torch
from torch.utils.data import DataLoader

from flaky import flaky
from sparseml.pytorch.sparsification.pruning import OBSPruningModifier
from sparseml.pytorch.utils import tensor_sparsity
from sparseml.utils import FROM_PARAM_TOKEN
from tests.sparseml.pytorch.helpers import MLPDataset, MLPNet
from tests.sparseml.pytorch.sparsification.pruning.helpers import (
    pruning_modifier_serialization_vals_test,
)
from tests.sparseml.pytorch.sparsification.test_modifier import (
    ScheduledUpdateModifierTest,
    create_optim_adam,
)


from tests.sparseml.pytorch.helpers import (  # noqa isort:skip
    test_epoch,
    test_loss,
    test_steps_per_epoch,
)


def _get_loss_function():
    return lambda model_outputs, loss_target: torch.nn.functional.mse_loss(
        model_outputs[0], loss_target
    )


def _get_dataloader_builder(
    dataset_lambda,
    obs_batch_size,
    num_grads,
    num_epochs,
    update_frequency,
):
    def dataloader_builder(kwargs: Optional[Dict[str, Any]] = None):
        batch_size = kwargs["batch_size"] if kwargs else obs_batch_size
        data_length = int(
            obs_batch_size * num_grads * num_epochs * (1 / update_frequency) * 2
        )
        dataset = dataset_lambda(length=data_length)
        data_loader = DataLoader(dataset, batch_size=batch_size)

        for sample in data_loader:
            img, target = [t for t in sample]
            yield [img], {}, target

    return dataloader_builder


@flaky(max_runs=3, min_passes=2)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "modifier_lambda",
    [
        lambda: OBSPruningModifier(
            init_sparsity=0.5,
            final_sparsity=0.95,
            start_epoch=2.0,
            end_epoch=5.0,
            update_frequency=1.0,
            params=["re:.*weight"],
            inter_func="cubic",
            mask_type="unstructured",
            num_grads=8,
            damp=1e-7,
            fisher_block_size=50,
        ),
        lambda: OBSPruningModifier(
            init_sparsity=FROM_PARAM_TOKEN,
            final_sparsity=0.95,
            start_epoch=2.0,
            end_epoch=5.0,
            update_frequency=1.0,
            params=["re:.*weight"],
            inter_func="linear",
            mask_type="block4",
            fisher_block_size=64,
            damp=0.000001,
            num_grads=8,
            grad_sampler_kwargs={"batch_size": 8},
        ),
        lambda: OBSPruningModifier(
            params=["seq.fc1.weight", "seq.fc2.weight"],
            init_sparsity=0.5,
            final_sparsity=0.95,
            start_epoch=2.0,
            end_epoch=5.0,
            update_frequency=1.0,
            inter_func="cubic",
            num_grads=8,
            global_sparsity=True,
        ),
    ],
    scope="function",
)
@pytest.mark.parametrize(
    "model_lambda",
    [MLPNet],
)
@pytest.mark.parametrize(
    "optim_lambda",
    [create_optim_adam],
    scope="function",
)
class TestOBSPruningModifier(ScheduledUpdateModifierTest):
    def get_default_initialize_kwargs(self) -> Dict[str, Any]:
        # add default no-op grad_sampler for initialization on non lifecycle tests
        return {
            "grad_sampler": {
                "data_loader_builder": lambda *loader_args, **loader_kwargs: [],
                "loss_function": lambda *loss_args, **loss_kwargs: None,
            }
        }

    @pytest.mark.parametrize(
        "dataset_lambda, obs_batch_size",
        [(MLPDataset, 4)],
    )
    def test_lifecycle(
        self,
        modifier_lambda,
        model_lambda,
        optim_lambda,
        test_steps_per_epoch,  # noqa: F811
        dataset_lambda,
        obs_batch_size,
    ):
        modifier = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)
        grad_sampler = {
            "data_loader_builder": _get_dataloader_builder(
                dataset_lambda,
                obs_batch_size,
                modifier.num_grads,
                modifier.end_epoch - modifier.start_epoch + 1,
                modifier.update_frequency,
            ),
            "loss_function": _get_loss_function(),
        }

        self.initialize_helper(modifier, model, grad_sampler=grad_sampler)
        if modifier.start_epoch > 0:
            assert modifier.applied_sparsity is None
        assert modifier._mask_creator == modifier._module_masks._mask_creator

        # check sparsity is not set before
        for epoch in range(int(modifier.start_epoch)):
            assert not modifier.update_ready(epoch, test_steps_per_epoch)
            assert modifier.applied_sparsity is None

        epoch = int(modifier.start_epoch)
        assert modifier.update_ready(epoch, test_steps_per_epoch)
        modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)

        applied_sparsities = modifier.applied_sparsity
        if not isinstance(applied_sparsities, list):
            applied_sparsities = [applied_sparsities]

        if not isinstance(modifier.init_sparsity, str):
            assert all(
                applied_sparsity == modifier.init_sparsity
                for applied_sparsity in applied_sparsities
            )
        else:
            assert len(modifier._init_sparsity) == len(modifier.module_masks.layers)
            for idx, param in enumerate(modifier.module_masks.params_data):
                assert modifier._init_sparsity[idx] == tensor_sparsity(param).item()

        last_sparsities = applied_sparsities

        # check forward pass
        input_shape = model_lambda.layer_descs()[0].input_size
        test_batch = torch.randn(10, *input_shape)
        _ = model(test_batch)

        while epoch < modifier.end_epoch - modifier.update_frequency:
            epoch += modifier.update_frequency
            assert modifier.update_ready(epoch, test_steps_per_epoch)
            modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)

            applied_sparsities = modifier.applied_sparsity
            if not isinstance(applied_sparsities, list):
                applied_sparsities = [applied_sparsities]

            assert all(
                applied_sparsity > last_sparsity
                for applied_sparsity, last_sparsity in zip(
                    applied_sparsities, last_sparsities
                )
            )

            last_sparsities = applied_sparsities

        _ = model(test_batch)  # check forward pass
        epoch = int(modifier.end_epoch)
        assert modifier.update_ready(epoch, test_steps_per_epoch)
        modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)

        def _test_final_sparsity_applied():
            final_sparsities = (
                [modifier.final_sparsity]
                if isinstance(modifier.final_sparsity, float)
                else modifier.final_sparsity
            )
            assert all(
                sparsity in final_sparsities for sparsity in modifier.applied_sparsity
            )

        _test_final_sparsity_applied()

        for epoch in range(int(modifier.end_epoch) + 1, int(modifier.end_epoch) + 6):
            assert not modifier.update_ready(epoch, test_steps_per_epoch)
            _test_final_sparsity_applied()

    @pytest.mark.parametrize(
        "dataset_lambda, obs_batch_size",
        [(MLPDataset, 4)],
    )
    def test_scheduled_update(
        self,
        modifier_lambda,
        model_lambda,
        optim_lambda,
        test_epoch,  # noqa: F811
        test_steps_per_epoch,  # noqa: F811
        dataset_lambda,
        obs_batch_size,
    ):
        modifier = modifier_lambda()
        grad_sampler = {
            "data_loader_builder": _get_dataloader_builder(
                dataset_lambda,
                obs_batch_size,
                modifier.num_grads,
                modifier.end_epoch - modifier.start_epoch + 1,
                modifier.update_frequency,
            ),
            "loss_function": _get_loss_function(),
        }

        super().test_scheduled_update(
            modifier_lambda,
            model_lambda,
            optim_lambda,
            test_epoch,
            test_steps_per_epoch,
            grad_sampler=grad_sampler,
        )


@pytest.mark.parametrize(
    "params,init_sparsity,final_sparsity",
    [
        (["re:.*weight"], 0.05, 0.8),
        (
            [],
            0.05,
            {0.7: ["param1"], 0.8: ["param2", "param3"], 0.9: ["param4", "param5"]},
        ),
        (["re:.*weight"], FROM_PARAM_TOKEN, 0.8),
        (
            [],
            FROM_PARAM_TOKEN,
            {0.7: ["param1"], 0.8: ["param2", "param3"], 0.9: ["param4", "param5"]},
        ),
    ],
)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_obs_pruning_yaml(params, init_sparsity, final_sparsity):
    start_epoch = 5.0
    end_epoch = 15.0
    update_frequency = 1.0
    inter_func = "cubic"
    global_sparsity = False
    num_grads = 64
    damp = 0.000001
    fisher_block_size = 20
    mask_type = "block4"
    batch_size = 4
    yaml_str = f"""
    !OBSPruningModifier
        init_sparsity: {init_sparsity}
        final_sparsity: {final_sparsity}
        start_epoch: {start_epoch}
        end_epoch: {end_epoch}
        update_frequency: {update_frequency}
        params: {params}
        inter_func: {inter_func}
        global_sparsity: {global_sparsity}
        mask_type: {mask_type}
        num_grads: {num_grads}
        damp: {damp}
        fisher_block_size: {fisher_block_size}
        grad_sampler_kwargs:
            batch_size: {batch_size}
    """
    yaml_modifier = OBSPruningModifier.load_obj(yaml_str)
    serialized_modifier = OBSPruningModifier.load_obj(
        str(yaml_modifier)
    )  # type: OBSPruningModifier
    obj_modifier = OBSPruningModifier(
        init_sparsity=init_sparsity,
        final_sparsity=final_sparsity,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        update_frequency=update_frequency,
        params=params,
        inter_func=inter_func,
        global_sparsity=global_sparsity,
        mask_type=mask_type,
        num_grads=num_grads,
        damp=damp,
        fisher_block_size=fisher_block_size,
        grad_sampler_kwargs={
            "batch_size": batch_size,
        },
    )
    assert isinstance(yaml_modifier, OBSPruningModifier)
    pruning_modifier_serialization_vals_test(
        yaml_modifier, serialized_modifier, obj_modifier
    )
    assert (
        str(yaml_modifier.global_sparsity)
        == str(serialized_modifier.global_sparsity)
        == str(obj_modifier.global_sparsity)
    )
    assert (
        str(yaml_modifier._num_grads)
        == str(serialized_modifier._num_grads)
        == str(obj_modifier._num_grads)
    )
    assert (
        str(yaml_modifier._damp)
        == str(serialized_modifier._damp)
        == str(obj_modifier._damp)
    )
    assert (
        str(yaml_modifier._fisher_block_size)
        == str(serialized_modifier._fisher_block_size)
        == str(obj_modifier._fisher_block_size)
    )
    assert (
        str(yaml_modifier.mask_type)
        == str(serialized_modifier.mask_type)
        == str(obj_modifier.mask_type)
    )
    assert (
        str(yaml_modifier._grad_sampler_kwargs)
        == str(serialized_modifier._grad_sampler_kwargs)
        == str(obj_modifier._grad_sampler_kwargs)
    )
