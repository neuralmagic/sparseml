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
from torch.nn import Conv2d, Linear

from sparseml.pytorch.optim import QuantizationModifier
from tests.sparseml.pytorch.helpers import LinearNet, create_optim_sgd
from tests.sparseml.pytorch.optim.test_modifier import ScheduledModifierTest


from tests.sparseml.pytorch.helpers import (  # noqa isort:skip
    test_epoch,
    test_loss,
    test_steps_per_epoch,
)

try:
    from torch import quantization as torch_quantization
except Exception:
    torch_quantization = None


QUANTIZATION_MODIFIERS = [
    lambda: QuantizationModifier(
        start_epoch=0.0,
        disable_quantization_observer_epoch=2,
        freeze_bn_stats_epoch=3.0,
    ),
    lambda: QuantizationModifier(start_epoch=2.0, submodules=["seq"]),
    lambda: QuantizationModifier(start_epoch=2.0, submodules=["seq.fc1"]),
    lambda: QuantizationModifier(
        start_epoch=2.0, submodules=["seq.fc1", "seq.block1.fc2"]
    ),
    lambda: QuantizationModifier(
        start_epoch=2.0,
        submodules=["seq.fc1", "seq.block1.fc2"],
        reduce_range=True,
    ),
]


def _is_valid_submodule(module_name, submodule_names):
    return module_name in submodule_names or any(
        module_name.startswith(name) for name in submodule_names
    )


def _is_quantiable_module(module):
    return isinstance(module, (Conv2d, Linear))


def _test_quantizable_module(module, qat_expected, reduce_range):
    if qat_expected:
        assert hasattr(module, "qconfig") and module.qconfig is not None
        assert hasattr(module, "weight_fake_quant") and (
            module.weight_fake_quant is not None
        )
        assert hasattr(module, "activation_post_process") and (
            module.activation_post_process is not None
        )
        assert module.qconfig.activation.p.keywords["reduce_range"] == reduce_range
    else:
        assert not hasattr(module, "qconfig") or module.qconfig is None
        assert not hasattr(module, "weight_fake_quant")
        assert not hasattr(module, "activation_post_process")


def _test_qat_applied(modifier, model):
    # test quantization mods are applied
    if not modifier.submodules or modifier.submodules == [""]:
        assert hasattr(model, "qconfig") and model.qconfig is not None
        submodules = [""]
        for module in model.modules():
            if _is_quantiable_module(module):
                _test_quantizable_module(module, True, modifier.reduce_range)
    else:
        assert not hasattr(model, "qconfig") or model.qconfig is None
        submodules = modifier.submodules
    # check qconfig propagation
    for name, module in model.named_modules():
        if _is_quantiable_module(module):
            _test_quantizable_module(
                module, _is_valid_submodule(name, submodules), modifier.reduce_range
            )


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_QUANT_TESTS", False),
    reason="Skipping pytorch torch quantization tests",
)
@pytest.mark.skipif(
    torch_quantization is None,
    reason="torch quantization not available",
)
@pytest.mark.parametrize("modifier_lambda", QUANTIZATION_MODIFIERS, scope="function")
@pytest.mark.parametrize("model_lambda", [LinearNet], scope="function")
@pytest.mark.parametrize("optim_lambda", [create_optim_sgd], scope="function")
class TestQuantizationModifierImpl(ScheduledModifierTest):
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

        update_epochs = [modifier.start_epoch]
        if modifier.disable_quantization_observer_epoch is not None:
            update_epochs.append(modifier.disable_quantization_observer_epoch)
        if modifier.freeze_bn_stats_epoch is not None:
            update_epochs.append(modifier.freeze_bn_stats_epoch)
        for epoch in update_epochs:
            assert modifier.update_ready(epoch, test_steps_per_epoch)
        # test update ready is still true after start epoch
        # even if quantization has not been applied yet
        assert modifier.update_ready(modifier.start_epoch + 0.1, test_steps_per_epoch)

        # test QAT setup
        if modifier.start_epoch > 0:
            for module in model.modules():
                assert not hasattr(module, "qconfig") or module.qconfig is None
        else:
            # QAT should be applied
            _test_qat_applied(modifier, model)

        modifier.scheduled_update(
            model, optimizer, modifier.start_epoch, test_steps_per_epoch
        )

        # test update ready is False after start epoch is applied, before diable epochs
        if (
            len(update_epochs) == 1
            or min(update_epochs[1:]) <= modifier.start_epoch + 1
        ):
            # test epochs in 0.1 intervals
            for epoch_interval in range(10):
                epoch_interval *= 0.1
                epoch = modifier.start_epoch + 0.1 * epoch_interval
                assert not modifier.update_ready(epoch, test_steps_per_epoch)

        _test_qat_applied(modifier, model)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_QUANT_TESTS", False),
    reason="Skipping pytorch torch quantization tests",
)
@pytest.mark.skipif(
    torch_quantization is None,
    reason="torch quantization not available",
)
def test_quantization_modifier_yaml():
    start_epoch = 0.0
    submodules = ["block.0", "block.2"]
    model_fuse_fn_name = "fuse_module"
    disable_quantization_observer_epoch = 2.0
    freeze_bn_stats_epoch = 3.0
    quantize_embeddings = False
    reduce_range = True
    yaml_str = f"""
        !QuantizationModifier
            start_epoch: {start_epoch}
            submodules: {submodules}
            model_fuse_fn_name: {model_fuse_fn_name}
            disable_quantization_observer_epoch: {disable_quantization_observer_epoch}
            freeze_bn_stats_epoch: {freeze_bn_stats_epoch}
            quantize_embeddings: {quantize_embeddings}
            reduce_range: {reduce_range}
        """
    yaml_modifier = QuantizationModifier.load_obj(
        yaml_str
    )  # type: QuantizationModifier
    serialized_modifier = QuantizationModifier.load_obj(
        str(yaml_modifier)
    )  # type: QuantizationModifier
    obj_modifier = QuantizationModifier(
        start_epoch=start_epoch,
        submodules=submodules,
        model_fuse_fn_name=model_fuse_fn_name,
        disable_quantization_observer_epoch=disable_quantization_observer_epoch,
        freeze_bn_stats_epoch=freeze_bn_stats_epoch,
        quantize_embeddings=quantize_embeddings,
        reduce_range=reduce_range,
    )

    assert isinstance(yaml_modifier, QuantizationModifier)
    assert (
        yaml_modifier.start_epoch
        == serialized_modifier.start_epoch
        == obj_modifier.start_epoch
    )
    assert (
        sorted(yaml_modifier.submodules)
        == sorted(serialized_modifier.submodules)
        == sorted(obj_modifier.submodules)
    )
    assert (
        yaml_modifier.model_fuse_fn_name
        == serialized_modifier.model_fuse_fn_name
        == obj_modifier.model_fuse_fn_name
    )
    assert (
        yaml_modifier.disable_quantization_observer_epoch
        == serialized_modifier.disable_quantization_observer_epoch
        == obj_modifier.disable_quantization_observer_epoch
    )
    assert (
        yaml_modifier.freeze_bn_stats_epoch
        == serialized_modifier.freeze_bn_stats_epoch
        == obj_modifier.freeze_bn_stats_epoch
    )
    assert (
        yaml_modifier.quantize_embeddings
        == serialized_modifier.quantize_embeddings
        == obj_modifier.quantize_embeddings
    )
    assert (
        yaml_modifier.reduce_range
        == serialized_modifier.reduce_range
        == obj_modifier.reduce_range
    )
