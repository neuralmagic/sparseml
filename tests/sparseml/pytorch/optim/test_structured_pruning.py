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


import tempfile

import pytest
import torch

from sparseml.onnx.optim import get_param_structured_pruning_group_dependencies
from sparseml.pytorch.models import mobilenet, resnet50
from sparseml.pytorch.optim import StructuredPruningModifier
from sparseml.pytorch.utils import export_onnx


def _get_param_shapes(module):
    return {name: param.shape for name, param in module.named_parameters()}


@pytest.mark.parametrize(
    "model_lambda,structure_type,sparsity,ignore_params",
    [
        (mobilenet, "channel", 0.3, ["input.conv.weight"]),
        (resnet50, "filter", 0.2, ["classifier.fc.weight"]),
    ],
)
def test_structured_pruning_one_shot_e2e(
    model_lambda, structure_type, sparsity, ignore_params
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

    modifier = StructuredPruningModifier(
        param_group_dependency_map=param_group_dependency_map,
        init_sparsity=sparsity,
        final_sparsity=sparsity,
        start_epoch=0,
        end_epoch=1,
        update_frequency=1,
        params=params,
        mask_type=structure_type,
    )

    # track shapes before pruning
    init_param_shapes = _get_param_shapes(module)

    # prune and compress
    modifier.apply(module)

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
