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

from sparseml.pytorch.sparsification.quantization.modifier_quantization import (
    QuantizationModifier,
)
from sparseml.pytorch.sparsification.quantization.quantize import QuantizationScheme
from tests.sparseml.pytorch.helpers import ConvNet, LinearNet, create_optim_sgd
from tests.sparseml.pytorch.sparsification.test_modifier import ScheduledModifierTest


from tests.sparseml.pytorch.helpers import (  # noqa isort:skip
    test_epoch,
    test_loss,
    test_steps_per_epoch,
)


try:
    from torch import quantization as torch_quantization
except Exception:
    torch_quantization = None


def _assert_qconfigs_equal(qconfig_1, qconfig_2):
    def _assert_observers_eq(observer_1, observer_2):
        assert type(observer_1).__name__ == type(observer_2).__name__

        if hasattr(observer_1, "p"):
            # assume observer is a partial, test by properties dict
            assert observer_1.p.keywords == observer_2.p.keywords
        else:
            # default to plain `==`
            assert observer_1 == observer_2

    # check activation and weight observers
    _assert_observers_eq(qconfig_1.activation, qconfig_2.activation)
    _assert_observers_eq(qconfig_1.weight, qconfig_2.weight)


def _test_quantized_module(module):
    quantization_scheme = getattr(module, "quantization_scheme", None)
    qconfig = getattr(module, "qconfig", None)
    assert quantization_scheme is not None
    assert qconfig is not None

    is_quant_wrapper = isinstance(module, torch_quantization.QuantWrapper)

    expected_qconfig = (
        quantization_scheme.get_wrapper_qconfig()
        if is_quant_wrapper
        else quantization_scheme.get_qconfig()
    )

    _assert_qconfigs_equal(qconfig, expected_qconfig)

    if is_quant_wrapper:
        assert hasattr(module.quant, "activation_post_process")
        assert isinstance(
            module.quant.activation_post_process, torch_quantization.FakeQuantize
        )
        # test wrapped module
        _test_quantized_module(module.module)


def _test_qat_applied(modifier, model):
    assert modifier._qat_enabled

    # TODO: update to test against expected target modules from modifier
    # for now, just test QuantWrappers are as expected
    quant_wrappers = [
        mod
        for mod in model.modules()
        if isinstance(mod, torch_quantization.QuantWrapper)
    ]
    assert len(quant_wrappers) > 0

    for module in model.modules():
        if isinstance(module, torch_quantization.QuantWrapper):
            _test_quantized_module(module)


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
@pytest.mark.parametrize(
    "modifier_lambda,model_lambda",
    [
        (lambda: QuantizationModifier(start_epoch=0.0), LinearNet),
        (
            lambda: QuantizationModifier(
                start_epoch=0.0, default_scheme=QuantizationScheme(weights=None)
            ),
            LinearNet,
        ),
        (lambda: QuantizationModifier(start_epoch=0.0), ConvNet),
        (lambda: QuantizationModifier(start_epoch=2.0), ConvNet),
    ],
    scope="function",
)
@pytest.mark.parametrize("optim_lambda", [create_optim_sgd], scope="function")
class TestQuantizationModifierImpl(ScheduledModifierTest):
    def test_lifecycle(
        self,
        modifier_lambda,
        model_lambda,
        optim_lambda,
        test_steps_per_epoch,  # noqa: F811,
    ):
        modifier = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)

        self.initialize_helper(modifier, model)

        for epoch in range(int(modifier.start_epoch)):
            assert not modifier.update_ready(epoch, test_steps_per_epoch)

        update_epochs = [modifier.start_epoch]
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
            pass

        modifier.scheduled_update(
            model, optimizer, modifier.start_epoch, test_steps_per_epoch
        )

        # test update ready is False after start epoch is applied, before disable epochs
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
    default_scheme = dict(
        input_activations=dict(num_bits=8, symmetric=True),
        weights=dict(num_bits=6, symmetric=False),
    )
    yaml_str = f"""
    !QuantizationModifier
        start_epoch: {start_epoch}
        default_scheme: {default_scheme}
    """
    yaml_modifier = QuantizationModifier.load_obj(
        yaml_str
    )  # type: QuantizationModifier
    serialized_modifier = QuantizationModifier.load_obj(
        str(yaml_modifier)
    )  # type: QuantizationModifier
    obj_modifier = QuantizationModifier(
        start_epoch=start_epoch,
        default_scheme=default_scheme,
    )

    assert isinstance(yaml_modifier, QuantizationModifier)
    assert (
        yaml_modifier.start_epoch
        == serialized_modifier.start_epoch
        == obj_modifier.start_epoch
    )
    assert (
        yaml_modifier.default_scheme
        == serialized_modifier.default_scheme
        == obj_modifier.default_scheme
    )
    assert isinstance(yaml_modifier.default_scheme, QuantizationScheme)
    assert isinstance(serialized_modifier.default_scheme, QuantizationScheme)
    assert isinstance(obj_modifier.default_scheme, QuantizationScheme)
