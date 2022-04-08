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
import tempfile

import pytest
import torch

from sparseml.onnx.optim import get_param_structured_pruning_group_dependencies
from sparseml.pytorch.models import mobilenet, resnet50
from sparseml.pytorch.sparsification import (
    LayerThinningModifier,
    StructuredPruningModifier,
)
from sparseml.pytorch.utils import export_onnx
from tests.sparseml.pytorch.helpers import LinearNet, create_optim_sgd
from tests.sparseml.pytorch.sparsification.test_modifier import ScheduledModifierTest


from tests.sparseml.pytorch.helpers import (  # noqa isort:skip
    test_epoch,
    test_loss,
    test_steps_per_epoch,
)


THINNING_MODIFIERS = [
    lambda: LayerThinningModifier(param_group_dependency_map={}, start_epoch=0.0),
    lambda: LayerThinningModifier(
        param_group_dependency_map={},
        start_epoch=0.0,
        update_epochs=[1.0, 2.0, 3.0],
    ),
]


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize("modifier_lambda", THINNING_MODIFIERS, scope="function")
@pytest.mark.parametrize("model_lambda", [LinearNet], scope="function")
@pytest.mark.parametrize("optim_lambda", [create_optim_sgd], scope="function")
class TestLayerThinningModifierImpl(ScheduledModifierTest):
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

        for epoch in range(int(modifier.start_epoch)):
            assert not modifier.update_ready(epoch, test_steps_per_epoch)

        assert modifier.update_ready(modifier.start_epoch, test_steps_per_epoch)

        modifier.scheduled_update(
            model, optimizer, modifier.start_epoch, test_steps_per_epoch
        )

        if not modifier.update_epochs:
            return

        max_epochs = max(modifier.update_epochs) + 1
        epoch = round(modifier.start_epoch + 0.1, 1)
        while epoch < max_epochs:
            update_ready = modifier.update_ready(epoch, test_steps_per_epoch)
            assert update_ready == (epoch in modifier.update_epochs)
            if update_ready:
                modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)
            epoch = round(epoch + 0.1, 1)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_thinning_modifier_yaml():
    start_epoch = 0.0
    param_group_dependency_map = {"param1,param2": ["dep1", "dep2"]}
    structure_type = "filter"
    update_epochs = [1.0, 2.0]
    strict = False
    yaml_str = f"""
        !LayerThinningModifier
            start_epoch: {start_epoch}
            param_group_dependency_map: {param_group_dependency_map}
            structure_type: {structure_type}
            update_epochs: {update_epochs}
            strict: {strict}
        """
    yaml_modifier = LayerThinningModifier.load_obj(
        yaml_str
    )  # type: LayerThinningModifier
    serialized_modifier = LayerThinningModifier.load_obj(
        str(yaml_modifier)
    )  # type: LayerThinningModifier
    obj_modifier = LayerThinningModifier(
        start_epoch=start_epoch,
        param_group_dependency_map=param_group_dependency_map,
        structure_type=structure_type,
        update_epochs=update_epochs,
        strict=strict,
    )

    assert isinstance(yaml_modifier, LayerThinningModifier)
    assert (
        yaml_modifier.start_epoch
        == serialized_modifier.start_epoch
        == obj_modifier.start_epoch
    )
    assert (
        sorted(yaml_modifier.param_group_dependency_map)
        == sorted(serialized_modifier.param_group_dependency_map)
        == sorted(obj_modifier.param_group_dependency_map)
    )
    assert (
        yaml_modifier.structure_type
        == serialized_modifier.structure_type
        == obj_modifier.structure_type
    )
    assert (
        sorted(yaml_modifier.update_epochs)
        == sorted(serialized_modifier.update_epochs)
        == sorted(obj_modifier.update_epochs)
    )
    assert yaml_modifier.strict == serialized_modifier.strict == obj_modifier.strict


def _get_param_shapes(module):
    return {name: param.shape for name, param in module.named_parameters()}


@pytest.mark.parametrize(
    "model_lambda,structure_type,sparsity,ignore_params",
    [
        (mobilenet, "channel", 0.3, ["input.conv.weight"]),
        (resnet50, "filter", 0.2, ["classifier.fc.weight"]),
    ],
)
@pytest.mark.parametrize(
    "strict",
    [True, False],
)
def test_structured_pruning_one_shot_e2e(
    model_lambda, structure_type, sparsity, ignore_params, strict
):
    # setup
    module = model_lambda()
    params = [
        name
        for name, _ in module.named_parameters()
        if ("conv" in name or "fc" in name) and "weight" in name
    ]

    tmp_fp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False).name
    export_onnx(module, torch.randn(1, 3, 224, 224), tmp_fp)

    param_group_dependency_map = get_param_structured_pruning_group_dependencies(
        tmp_fp, structure_type
    )

    for param_name in ignore_params:
        params_to_delete = [param_name]
        keys_to_delete = [k for k in param_group_dependency_map if param_name in k]

        for key in keys_to_delete:
            params_to_delete.extend(key.split(","))
            del [param_group_dependency_map[key]]

        for param_to_delete in params_to_delete:
            if param_to_delete in params:
                del params[params.index(param_to_delete)]

    # track shapes before pruning
    init_param_shapes = _get_param_shapes(module)

    param_groups = [key.split(",") for key in param_group_dependency_map]
    param_groups = [group for group in param_groups if len(group) > 1]
    pruning_modifier = StructuredPruningModifier(
        param_groups=param_groups if strict else [],
        init_sparsity=sparsity,
        final_sparsity=sparsity,
        start_epoch=0,
        end_epoch=1,
        update_frequency=1,
        params=params,
        mask_type=structure_type,
    )
    thinning_modifier = LayerThinningModifier(
        start_epoch=0.0,
        param_group_dependency_map=param_group_dependency_map,
        structure_type=structure_type,
        strict=strict,
    )

    # prune and thin
    pruning_modifier.apply(module)
    thinning_modifier.apply(module)

    # get updated shapes
    pruned_param_shapes = _get_param_shapes(module)

    # validate that update shapes have been pruned to target sparsity
    target_dim = 0 if structure_type == "filter" else 1

    for param_name in params:
        init_shape = init_param_shapes[param_name]

        if init_shape[target_dim] == 1:
            # DW Conv
            continue

        pruned_shape = pruned_param_shapes[param_name]
        applied_compression = 1 - float(pruned_shape[target_dim]) / float(
            init_shape[target_dim]
        )
        # allow a 0.05 room for error since pruned dims may be small
        assert abs(applied_compression - sparsity) < 5e-2
    # validate forward pass
    assert module(torch.randn(2, 3, 224, 224)) is not None
