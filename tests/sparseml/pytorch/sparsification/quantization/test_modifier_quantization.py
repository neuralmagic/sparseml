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
from torch.nn import Identity

from sparseml.pytorch.sparsification.quantization.helpers import QATWrapper
from sparseml.pytorch.sparsification.quantization.modifier_quantization import (
    QuantizationModifier,
)
from sparseml.pytorch.sparsification.quantization.quantization_scheme import (
    QuantizationScheme,
)
from sparseml.pytorch.sparsification.quantization.quantize import (
    is_qat_helper_module,
    is_quantizable_module,
)
from tests.sparseml.pytorch.helpers import (
    ConvNet,
    LinearNet,
    QATMatMulTestNet,
    create_optim_sgd,
)
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


def _test_quantized_module(base_model, modifier, module, name):
    # check quant scheme and configs are set
    quantization_scheme = getattr(module, "quantization_scheme", None)
    qconfig = getattr(module, "qconfig", None)
    assert quantization_scheme is not None
    assert qconfig is not None

    # if module type is overwritten in by scheme_overrides, check scheme set correctly
    module_type_name = module.__class__.__name__
    if module_type_name in modifier.scheme_overrides:
        expected_scheme = modifier.scheme_overrides[module_type_name]
        assert quantization_scheme == expected_scheme

    is_quant_wrapper = isinstance(module, torch_quantization.QuantWrapper)

    expected_qconfig = (
        quantization_scheme.get_wrapper_qconfig()
        if is_quant_wrapper
        else quantization_scheme.get_qconfig()
    )

    # check generated qconfig matches expected
    _assert_qconfigs_equal(qconfig, expected_qconfig)

    if is_quant_wrapper:
        # assert that activations are tracked by quant stub observer
        assert not hasattr(module, "activation_post_process")
        assert hasattr(module.quant, "activation_post_process")
        assert isinstance(
            module.quant.activation_post_process, torch_quantization.FakeQuantize
        )
    else:
        if quantization_scheme.input_activations:
            # assert that parent is a QuantWrapper to quantize input activations
            parent_module = base_model
            for layer in name.split(".")[:-1]:
                parent_module = getattr(parent_module, layer)
            assert isinstance(parent_module, torch_quantization.QuantWrapper)

        # all non wrapper modules targeted for quantization should have a post process
        # which is a FakeQuantize quantizing outputs, or an Identity if not
        assert hasattr(module, "activation_post_process")
        if quantization_scheme.output_activations is None:
            assert isinstance(module.activation_post_process, Identity)
        else:
            assert isinstance(
                module.activation_post_process, torch_quantization.FakeQuantize
            )


def _test_qat_wrapped_module(root_module, wrapped_module_name):
    parent_module = root_module
    for layer in wrapped_module_name.split(".")[:-1]:
        parent_module = getattr(parent_module, layer)
    assert isinstance(parent_module, QATWrapper)


def _test_qat_applied(modifier, model):
    assert modifier._qat_enabled

    for name, module in model.named_modules():
        if is_qat_helper_module(module):
            # skip helper modules
            continue

        is_target_submodule = not any(
            name.startswith(submodule_name) for submodule_name in modifier.ignore
        )
        is_included_module_type = any(
            module_type_name == module.__class__.__name__
            for module_type_name in modifier.scheme_overrides
        )
        is_quantizable = is_included_module_type or is_quantizable_module(
            module,
            exclude_module_types=modifier.ignore,
        )

        if is_target_submodule and is_quantizable:
            if getattr(module, "wrap_qat", False):
                _test_qat_wrapped_module(model, name)
            elif is_quantizable:
                # check each target module is quantized
                _test_quantized_module(model, modifier, module, name)
        else:
            # check all non-target modules are not quantized
            assert not hasattr(module, "quantization_scheme")
            assert not hasattr(module, "qconfig")


def _test_freeze_bn_stats_observer_applied(modifier, epoch):
    if modifier.disable_quantization_observer_epoch is not None and (
        epoch >= modifier.disable_quantization_observer_epoch
    ):
        assert modifier._quantization_observer_disabled
    else:
        assert not modifier._quantization_observer_disabled
    if modifier.freeze_bn_stats_epoch is not None and (
        epoch >= modifier.freeze_bn_stats_epoch
    ):
        assert modifier._bn_stats_frozen
    else:
        assert not modifier._bn_stats_frozen


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
                start_epoch=0.0, scheme=QuantizationScheme(weights=None)
            ),
            LinearNet,
        ),
        (
            lambda: QuantizationModifier(
                start_epoch=0.0,
                scheme_overrides=dict(seq="default"),
                freeze_bn_stats_epoch=2.0,
                disable_quantization_observer_epoch=3.0,
            ),
            LinearNet,
        ),
        (
            lambda: QuantizationModifier(
                start_epoch=1.0,
                scheme_overrides=dict(Linear=QuantizationScheme(weights=None)),
                ignore=["seq.fc1"],
            ),
            LinearNet,
        ),
        (
            lambda: QuantizationModifier(start_epoch=0.0, ignore=["Linear"]),
            LinearNet,
        ),
        (lambda: QuantizationModifier(start_epoch=0.0), ConvNet),
        (
            lambda: QuantizationModifier(
                start_epoch=2.0, scheme_overrides=dict(mlp="deepsparse")
            ),
            ConvNet,
        ),
        (
            lambda: QuantizationModifier(
                start_epoch=2.0,
                scheme_overrides=dict(Conv2d=QuantizationScheme(weights=None)),
                freeze_bn_stats_epoch=2.5,
                disable_quantization_observer_epoch=2.2,
                ignore=["mlp"],
            ),
            ConvNet,
        ),
        (
            lambda: QuantizationModifier(
                start_epoch=0.0,
                scheme_overrides=dict(
                    seq="tensorrt",
                    mlp=QuantizationScheme(weights=None),
                    Conv2d=QuantizationScheme(output_activations={}),
                ),
                ignore=["ReLU", "mlp"],
            ),
            ConvNet,
        ),
        # test wrap_qat feature
        (lambda: QuantizationModifier(start_epoch=0.0), QATMatMulTestNet),
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

        _test_freeze_bn_stats_observer_applied(modifier, 0.0)
        for epoch in range(int(modifier.start_epoch)):
            assert not modifier.update_ready(epoch, test_steps_per_epoch)

        update_epochs = [modifier.start_epoch]
        if modifier.disable_quantization_observer_epoch is not None:
            update_epochs.append(modifier.disable_quantization_observer_epoch)
        if modifier.freeze_bn_stats_epoch is not None:
            update_epochs.append(modifier.freeze_bn_stats_epoch)
        update_epochs.sort()
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

        for update_epoch in update_epochs:
            modifier.scheduled_update(
                model, optimizer, update_epoch, test_steps_per_epoch
            )
            _test_freeze_bn_stats_observer_applied(modifier, update_epoch)

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
    scheme = dict(
        input_activations=dict(num_bits=8, symmetric=True),
        weights=dict(num_bits=6, symmetric=False),
    )
    scheme_overrides = dict(
        feature_extractor="deepsparse",
        classifier=dict(
            input_activations=dict(num_bits=8, symmetric=True),
            weights=None,
        ),
        Linear=dict(output_activations=dict(symmetric=False)),
    )
    ignore = ["LayerNorm", "Tanh"]
    disable_quantization_observer_epoch = 2.0
    freeze_bn_stats_epoch = 3.0
    num_calibration_steps = 1000
    model_fuse_fn_name = "custom_fuse_fn"
    model_fuse_fn_kwargs = dict(inplace=True)
    strict = False

    yaml_str = f"""
    !QuantizationModifier
        start_epoch: {start_epoch}
        scheme: {scheme}
        scheme_overrides: {scheme_overrides}
        ignore: {ignore}
        disable_quantization_observer_epoch: {disable_quantization_observer_epoch}
        freeze_bn_stats_epoch: {freeze_bn_stats_epoch}
        num_calibration_steps: {num_calibration_steps}
        model_fuse_fn_name: {model_fuse_fn_name}
        model_fuse_fn_kwargs: {model_fuse_fn_kwargs}
        strict: {strict}
    """
    yaml_modifier = QuantizationModifier.load_obj(
        yaml_str
    )  # type: QuantizationModifier
    serialized_modifier = QuantizationModifier.load_obj(
        str(yaml_modifier)
    )  # type: QuantizationModifier
    obj_modifier = QuantizationModifier(
        start_epoch=start_epoch,
        scheme=scheme,
        scheme_overrides=scheme_overrides,
        ignore=ignore,
        disable_quantization_observer_epoch=disable_quantization_observer_epoch,
        freeze_bn_stats_epoch=freeze_bn_stats_epoch,
        num_calibration_steps=num_calibration_steps,
        model_fuse_fn_name=model_fuse_fn_name,
        model_fuse_fn_kwargs=model_fuse_fn_kwargs,
        strict=strict,
    )

    assert isinstance(yaml_modifier, QuantizationModifier)
    assert (
        yaml_modifier.start_epoch
        == serialized_modifier.start_epoch
        == obj_modifier.start_epoch
    )
    assert yaml_modifier.scheme == serialized_modifier.scheme == obj_modifier.scheme
    assert isinstance(yaml_modifier.scheme, QuantizationScheme)
    assert isinstance(serialized_modifier.scheme, QuantizationScheme)
    assert isinstance(obj_modifier.scheme, QuantizationScheme)
    assert (
        yaml_modifier.scheme_overrides
        == serialized_modifier.scheme_overrides
        == obj_modifier.scheme_overrides
    )
    assert yaml_modifier.ignore == serialized_modifier.ignore == obj_modifier.ignore
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
        yaml_modifier.num_calibration_steps
        == serialized_modifier.num_calibration_steps
        == obj_modifier.num_calibration_steps
    )
    assert (
        yaml_modifier.model_fuse_fn_name
        == serialized_modifier.model_fuse_fn_name
        == obj_modifier.model_fuse_fn_name
    )
    assert (
        yaml_modifier.model_fuse_fn_kwargs
        == serialized_modifier.model_fuse_fn_kwargs
        == obj_modifier.model_fuse_fn_kwargs
    )
    assert yaml_modifier.strict == serialized_modifier.strict == obj_modifier.strict


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
    "modifier_lambda,unmatched_values",
    [
        (
            lambda: QuantizationModifier(
                scheme_overrides=dict(mlp={}, submodule_1={}, Conv2d={}, Embedding={}),
            ),
            ["submodule_1", "Embedding"],
        ),
        (
            lambda: QuantizationModifier(
                ignore=dict(mlp={}, submodule_1={}, Conv2d={}, Embedding={}),
            ),
            ["submodule_1", "Embedding"],
        ),
        (
            lambda: QuantizationModifier(
                ignore=dict(mlp={}, submodule_1={}),
                scheme_overrides=dict(Conv2d={}, Embedding={}),
            ),
            ["Embedding"],
        ),
    ],
)
def test_strict_raises_expected_error(modifier_lambda, unmatched_values):
    model = ConvNet()
    modifier = modifier_lambda()

    with pytest.raises(ValueError) as value_error:
        modifier.apply(model)
    assert f"unmatched values: {unmatched_values}" in value_error.value.args[0]
