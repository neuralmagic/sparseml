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

from sparseml.pytorch.sparsification import (
    GradualParamModifier,
    SetParamModifier,
    TrainableParamsModifier,
)
from sparseml.pytorch.utils import (
    any_str_or_regex_matches_param_name,
    get_named_layers_and_params_by_regex,
)
from sparseml.utils import ALL_TOKEN
from tests.sparseml.pytorch.helpers import LinearNet, create_optim_sgd
from tests.sparseml.pytorch.sparsification.test_modifier import (
    ScheduledModifierTest,
    ScheduledUpdateModifierTest,
)


from tests.sparseml.pytorch.helpers import (  # noqa isort:skip
    test_epoch,
    test_loss,
    test_steps_per_epoch,
)


MODEL_TEST_LAYER_INDEX = 3


##############################
#
# TrainableParamsModifier tests
#
##############################
TRAINABLE_MODIFIERS = [
    lambda: TrainableParamsModifier(params=ALL_TOKEN, trainable=False, start_epoch=0.0),
    lambda: TrainableParamsModifier(params=ALL_TOKEN, trainable=True, start_epoch=0.0),
    lambda: TrainableParamsModifier(
        params=ALL_TOKEN,
        trainable=False,
        start_epoch=0.0,
        end_epoch=15.0,
    ),
    lambda: TrainableParamsModifier(
        params=ALL_TOKEN,
        trainable=True,
        start_epoch=10.0,
        end_epoch=25.0,
    ),
    lambda: TrainableParamsModifier(
        params=["re:.*weight"],
        trainable=False,
        start_epoch=10.0,
    ),
]


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize("modifier_lambda", TRAINABLE_MODIFIERS, scope="function")
@pytest.mark.parametrize("model_lambda", [LinearNet], scope="function")
@pytest.mark.parametrize("optim_lambda", [create_optim_sgd], scope="function")
class TestTrainableParamsModifierImpl(ScheduledModifierTest):
    def test_lifecycle(
        self, modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
    ):
        modifier = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)

        for param in model.parameters():
            param.requires_grad = not modifier.trainable

        self.initialize_helper(modifier, model)
        for param in model.parameters():
            if modifier.start_epoch == 0.0 or modifier.start_epoch == -1.0:
                # modifier starts at init (e.g. one-shot)
                assert param.requires_grad == modifier.trainable
            else:
                assert param.requires_grad != modifier.trainable

        for epoch in range(int(modifier.start_epoch)):
            assert not modifier.update_ready(epoch, test_steps_per_epoch)
            for param in model.parameters():
                assert param.requires_grad != modifier.trainable

        epoch = int(modifier.start_epoch)
        assert modifier.update_ready(epoch, test_steps_per_epoch)
        modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)
        for name, param in model.named_parameters():
            if modifier.params == ALL_TOKEN or any_str_or_regex_matches_param_name(
                name, modifier.params
            ):
                assert param.requires_grad == modifier.trainable
            else:
                assert param.requires_grad != modifier.trainable

        test_set_epochs = [
            e
            for e in range(
                epoch + 1,
                int(modifier.end_epoch) if modifier.end_epoch > 0 else epoch + 6,
            )
        ]
        for epoch in test_set_epochs:
            assert not modifier.update_ready(epoch, test_steps_per_epoch)

            if modifier.end_epoch > 0:
                for name, param in model.named_parameters():
                    if (
                        modifier.params == ALL_TOKEN
                        or any_str_or_regex_matches_param_name(name, modifier.params)
                    ):
                        assert param.requires_grad == modifier.trainable
                    else:
                        assert param.requires_grad != modifier.trainable

        if modifier.end_epoch > 0:
            epoch = int(modifier.end_epoch)
            assert modifier.update_ready(epoch, test_set_epochs)
            modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)
            for name, param in model.named_parameters():
                assert param.requires_grad != modifier.trainable

            for epoch in range(
                int(modifier.end_epoch) + 1, int(modifier.end_epoch) + 6
            ):
                assert not modifier.update_ready(epoch, test_steps_per_epoch)

                if modifier.end_epoch > 0:
                    for name, param in model.named_parameters():
                        assert param.requires_grad != modifier.trainable


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_trainable_params_yaml():
    params = ALL_TOKEN
    trainable = False
    params_strict = False
    start_epoch = 10.0
    end_epoch = 20.0
    yaml_str = """
    !TrainableParamsModifier
        params: {params}
        trainable: {trainable}
        params_strict: {params_strict}
        start_epoch: {start_epoch}
        end_epoch: {end_epoch}
    """.format(
        params=params,
        trainable=trainable,
        params_strict=params_strict,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
    )
    yaml_modifier = TrainableParamsModifier.load_obj(
        yaml_str
    )  # type: TrainableParamsModifier
    serialized_modifier = TrainableParamsModifier.load_obj(
        str(yaml_modifier)
    )  # type: TrainableParamsModifier
    obj_modifier = TrainableParamsModifier(
        params=params,
        trainable=trainable,
        params_strict=params_strict,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
    )

    assert isinstance(yaml_modifier, TrainableParamsModifier)
    assert yaml_modifier.params == serialized_modifier.params == obj_modifier.params
    assert (
        yaml_modifier.trainable
        == serialized_modifier.trainable
        == obj_modifier.trainable
    )
    assert (
        yaml_modifier.params_strict
        == serialized_modifier.params_strict
        == obj_modifier.params_strict
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


##############################
#
# SetParamModifier tests
#
##############################
assert LinearNet.layer_descs()[MODEL_TEST_LAYER_INDEX].bias  # testing bias below
SET_PARAM_MOD_VAL = [
    0 for _ in range(LinearNet.layer_descs()[MODEL_TEST_LAYER_INDEX].output_size[0])
]
SET_PARAM_MOD_PARAMS = [
    "{}.{}".format(LinearNet.layer_descs()[MODEL_TEST_LAYER_INDEX].name, "bias")
]
SET_PARAM_MODIFIERS = [
    lambda: SetParamModifier(
        params=SET_PARAM_MOD_PARAMS,
        val=SET_PARAM_MOD_VAL,
        start_epoch=0.0,
    ),
    lambda: SetParamModifier(
        params=SET_PARAM_MOD_PARAMS,
        val=SET_PARAM_MOD_VAL,
        start_epoch=10.0,
    ),
]


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize("modifier_lambda", SET_PARAM_MODIFIERS, scope="function")
@pytest.mark.parametrize("model_lambda", [LinearNet], scope="function")
@pytest.mark.parametrize("optim_lambda", [create_optim_sgd], scope="function")
class TestSetParamModifierImpl(ScheduledModifierTest):
    def test_lifecycle(
        self, modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
    ):
        modifier = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)
        self._lifecycle_helper(modifier, model, optimizer)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires cuda availability"
    )
    def test_lifecycle_cuda(
        self, modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
    ):
        modifier = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)
        model = model.to("cuda")
        self._lifecycle_helper(modifier, model, optimizer)

    def _lifecycle_helper(self, modifier, model, optimizer):
        param_regex = modifier.params if modifier.params != ALL_TOKEN else ["re:.*"]
        named_layers_and_params = get_named_layers_and_params_by_regex(
            model, param_regex
        )
        _, _, _, param = named_layers_and_params[0]
        self.initialize_helper(modifier, model)
        for set_val, param_val in zip(modifier.val, param.data):
            assert set_val != param_val.cpu()

        for epoch in range(int(modifier.start_epoch)):
            assert not modifier.update_ready(epoch, test_steps_per_epoch)
            for set_val, param_val in zip(modifier.val, param.data):
                assert set_val != param_val.cpu()

        epoch = int(modifier.start_epoch)
        assert modifier.update_ready(epoch, test_steps_per_epoch)
        modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)
        for set_val, param_val in zip(modifier.val, param.data):
            assert set_val == param_val.cpu()

        for epoch in range(
            int(modifier.start_epoch) + 1, int(modifier.start_epoch) + 6
        ):
            assert not modifier.update_ready(epoch, test_steps_per_epoch)
            for set_val, param_val in zip(modifier.val, param.data):
                assert set_val == param_val.cpu()


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_set_param_yaml():
    params_strict = False
    start_epoch = 10.0
    yaml_str = """
    !SetParamModifier
        params: {SET_PARAM_MOD_PARAMS}
        val: {SET_PARAM_MOD_VAL}
        params_strict: {params_strict}
        start_epoch: {start_epoch}
    """.format(
        SET_PARAM_MOD_PARAMS=SET_PARAM_MOD_PARAMS,
        SET_PARAM_MOD_VAL=SET_PARAM_MOD_VAL,
        params_strict=params_strict,
        start_epoch=start_epoch,
    )
    yaml_modifier = SetParamModifier.load_obj(yaml_str)  # type: SetParamModifier
    serialized_modifier = SetParamModifier.load_obj(
        str(yaml_modifier)
    )  # type: SetParamModifier
    obj_modifier = SetParamModifier(
        params=SET_PARAM_MOD_PARAMS,
        val=SET_PARAM_MOD_VAL,
        params_strict=params_strict,
        start_epoch=start_epoch,
    )

    assert isinstance(yaml_modifier, SetParamModifier)
    assert yaml_modifier.params == serialized_modifier.params == obj_modifier.params
    assert yaml_modifier.val == obj_modifier.val == obj_modifier.val
    assert (
        yaml_modifier.params_strict
        == obj_modifier.params_strict
        == obj_modifier.params_strict
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


##############################
#
# GradualParamModifier tests
#
##############################
assert LinearNet.layer_descs()[MODEL_TEST_LAYER_INDEX].bias  # testing bias below
GRADUAL_PARAM_MOD_INIT_VAL = [
    0.1 * (c + 1)
    for c in range(LinearNet.layer_descs()[MODEL_TEST_LAYER_INDEX].output_size[0])
]
GRADUAL_PARAM_MOD_FINAL_VAL = [
    1.0 * (c + 1)
    for c in range(LinearNet.layer_descs()[MODEL_TEST_LAYER_INDEX].output_size[0])
]
GRADUAL_PARAM_MOD_PARAMS = [
    "{}.{}".format(LinearNet.layer_descs()[MODEL_TEST_LAYER_INDEX].name, "bias")
]
GRADUAL_PARAM_MODIFIERS = [
    lambda: GradualParamModifier(
        params=GRADUAL_PARAM_MOD_PARAMS,
        init_val=GRADUAL_PARAM_MOD_INIT_VAL,
        final_val=GRADUAL_PARAM_MOD_FINAL_VAL,
        start_epoch=0.0,
        end_epoch=10.0,
        update_frequency=1.0,
        inter_func="linear",
        params_strict=True,
    ),
    lambda: GradualParamModifier(
        params=GRADUAL_PARAM_MOD_PARAMS,
        init_val=GRADUAL_PARAM_MOD_INIT_VAL,
        final_val=GRADUAL_PARAM_MOD_FINAL_VAL,
        start_epoch=10.0,
        end_epoch=20.0,
        update_frequency=1.0,
        inter_func="linear",
        params_strict=True,
    ),
]


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize("modifier_lambda", GRADUAL_PARAM_MODIFIERS, scope="function")
@pytest.mark.parametrize("model_lambda", [LinearNet], scope="function")
@pytest.mark.parametrize("optim_lambda", [create_optim_sgd], scope="function")
class TestGradualParamModifierImpl(ScheduledUpdateModifierTest):
    def test_lifecycle(
        self, modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
    ):
        modifier = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)
        self._lifecycle_helper(modifier, model, optimizer)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires cuda availability"
    )
    def test_lifecycle_cuda(
        self, modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
    ):
        modifier = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)
        model = model.to("cuda")
        self._lifecycle_helper(modifier, model, optimizer)

    def _lifecycle_helper(self, modifier, model, optimizer):
        param_regex = modifier.params if modifier.params != ALL_TOKEN else ["re:.*"]
        named_layers_and_params = get_named_layers_and_params_by_regex(
            model, param_regex
        )
        _, _, _, param = named_layers_and_params[0]
        self.initialize_helper(modifier, model)

        for set_val, param_val in zip(modifier.init_val, param.data):
            assert set_val != param_val.cpu()

        for epoch in range(int(modifier.start_epoch)):
            assert not modifier.update_ready(epoch, test_steps_per_epoch)
            for set_val, param_val in zip(modifier.init_val, param.data):
                assert set_val != param_val.cpu()

        epoch = int(modifier.start_epoch)
        assert modifier.update_ready(epoch, test_steps_per_epoch)
        modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)
        check_vals = []
        for set_val, param_val in zip(modifier.init_val, param.data):
            assert set_val == param_val.cpu()
            check_vals.append(param_val.cpu().clone())

        for epoch in range(int(modifier.start_epoch) + 1, int(modifier.end_epoch)):
            assert modifier.update_ready(epoch, test_steps_per_epoch)
            modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)

            for index, (set_val, param_val) in enumerate(zip(check_vals, param.data)):
                assert set_val < param_val.cpu()
                check_vals[index] = param_val.cpu().clone()

        epoch = int(modifier.end_epoch)
        assert modifier.update_ready(epoch, test_steps_per_epoch)
        modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)

        for epoch in range(int(modifier.end_epoch) + 1, int(modifier.end_epoch) + 6):
            for set_val, param_val in zip(modifier.final_val, param.data):
                assert set_val == param_val.cpu()

            assert not modifier.update_ready(epoch, test_steps_per_epoch)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_gradual_param_yaml():
    params_strict = False
    start_epoch = 10.0
    end_epoch = 20.0
    update_frequency = 1.0
    inter_func = "linear"
    yaml_str = """
    !GradualParamModifier
        params: {GRADUAL_PARAM_MOD_PARAMS}
        init_val: {GRADUAL_PARAM_MOD_INIT_VAL}
        final_val: {GRADUAL_PARAM_MOD_FINAL_VAL}
        start_epoch: {start_epoch}
        end_epoch: {end_epoch}
        update_frequency: {update_frequency}
        inter_func: {inter_func}
        params_strict: {params_strict}
    """.format(
        GRADUAL_PARAM_MOD_PARAMS=GRADUAL_PARAM_MOD_PARAMS,
        GRADUAL_PARAM_MOD_INIT_VAL=GRADUAL_PARAM_MOD_INIT_VAL,
        GRADUAL_PARAM_MOD_FINAL_VAL=GRADUAL_PARAM_MOD_FINAL_VAL,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        update_frequency=update_frequency,
        inter_func=inter_func,
        params_strict=params_strict,
    )
    yaml_modifier = GradualParamModifier.load_obj(
        yaml_str
    )  # type: GradualParamModifier
    serialized_modifier = GradualParamModifier.load_obj(
        str(yaml_modifier)
    )  # type: GradualParamModifier
    obj_modifier = GradualParamModifier(
        params=GRADUAL_PARAM_MOD_PARAMS,
        init_val=GRADUAL_PARAM_MOD_INIT_VAL,
        final_val=GRADUAL_PARAM_MOD_FINAL_VAL,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        update_frequency=update_frequency,
        inter_func=inter_func,
        params_strict=params_strict,
    )

    assert isinstance(yaml_modifier, GradualParamModifier)
    assert yaml_modifier.params == serialized_modifier.params == obj_modifier.params
    assert (
        yaml_modifier.init_val == serialized_modifier.init_val == obj_modifier.init_val
    )
    assert (
        yaml_modifier.final_val
        == serialized_modifier.final_val
        == obj_modifier.final_val
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
    assert (
        yaml_modifier.inter_func
        == serialized_modifier.inter_func
        == obj_modifier.inter_func
    )
    assert (
        yaml_modifier.params_strict
        == serialized_modifier.params_strict
        == obj_modifier.params_strict
    )
