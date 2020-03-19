import pytest

import torch

from neuralmagicML.utils import ALL_TOKEN
from neuralmagicML.pytorch.recal import (
    TrainableParamsModifier,
    SetParamModifier,
    GradualParamModifier,
)
from neuralmagicML.pytorch.utils import get_layer_param

from .test_modifier import (
    ScheduledModifierTest,
    ScheduledUpdateModifierTest,
    def_model,
    def_optim_sgd,
    DEFAULT_MODEL_LAYER,
    DEFAULT_MODEL_LAYER_PARAM,
    DEFAULT_MODEL_LAYER_PARAM_SIZE,
    test_epoch,
    test_steps_per_epoch,
    test_loss,
)


##############################
#
# TrainableParamsModifier tests
#
##############################
TRAINABLE_MODIFIERS = [
    lambda: TrainableParamsModifier(
        params=ALL_TOKEN, layers=ALL_TOKEN, trainable=False, start_epoch=0.0
    ),
    lambda: TrainableParamsModifier(
        params=ALL_TOKEN, layers=ALL_TOKEN, trainable=True, start_epoch=0.0
    ),
    lambda: TrainableParamsModifier(
        params=ALL_TOKEN,
        layers=ALL_TOKEN,
        trainable=False,
        start_epoch=0.0,
        end_epoch=15.0,
    ),
    lambda: TrainableParamsModifier(
        params=ALL_TOKEN,
        layers=ALL_TOKEN,
        trainable=True,
        start_epoch=10.0,
        end_epoch=25.0,
    ),
    lambda: TrainableParamsModifier(
        params=["weight"],
        layers=[DEFAULT_MODEL_LAYER],
        trainable=False,
        start_epoch=10.0,
    ),
]


@pytest.mark.parametrize("modifier_lambda", TRAINABLE_MODIFIERS, scope="function")
@pytest.mark.parametrize("model_lambda", [def_model], scope="function")
@pytest.mark.parametrize("optim_lambda", [def_optim_sgd], scope="function")
class TestTrainableParamsModifierImpl(ScheduledModifierTest):
    def test_lifecycle(
        self, modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
    ):
        modifier, model, optimizer = self.create_test_objs(
            modifier_lambda, model_lambda, optim_lambda
        )

        for param in model.parameters():
            param.requires_grad = not modifier.trainable

        self.initialize_helper(modifier, model, optimizer)
        for param in model.parameters():
            assert param.requires_grad != modifier.trainable

        for epoch in range(int(modifier.start_epoch)):
            assert not modifier.update_ready(epoch, test_steps_per_epoch)
            for param in model.parameters():
                assert param.requires_grad != modifier.trainable

        epoch = int(modifier.start_epoch)
        assert modifier.update_ready(epoch, test_steps_per_epoch)
        modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)
        for name, param in model.named_parameters():
            layer_name = ".".join(name.split(".")[:-1])
            param_name = name.split(".")[-1]

            if (modifier.layers == ALL_TOKEN or layer_name in modifier.layers) and (
                modifier.params == ALL_TOKEN or param_name in modifier.params
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
                    layer_name = ".".join(name.split(".")[:-1])
                    param_name = name.split(".")[-1]

                    if (
                        modifier.layers == ALL_TOKEN or layer_name in modifier.layers
                    ) and (
                        modifier.params == ALL_TOKEN or param_name in modifier.params
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


def test_set_lr_yaml():
    params = ALL_TOKEN
    layers = ALL_TOKEN
    trainable = False
    params_strict = False
    start_epoch = 10.0
    end_epoch = 20.0
    yaml_str = """
    !TrainableParamsModifier
        params: {}
        layers: {}
        trainable: {}
        params_strict: {}
        start_epoch: {}
        end_epoch: {}
    """.format(
        params, layers, trainable, params_strict, start_epoch, end_epoch
    )
    yaml_modifier = TrainableParamsModifier.load_obj(
        yaml_str
    )  # type: TrainableParamsModifier
    serialized_modifier = TrainableParamsModifier.load_obj(
        str(yaml_modifier)
    )  # type: TrainableParamsModifier
    obj_modifier = TrainableParamsModifier(
        params=params,
        layers=layers,
        trainable=trainable,
        params_strict=params_strict,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
    )

    assert isinstance(yaml_modifier, TrainableParamsModifier)
    assert yaml_modifier.params == serialized_modifier.params == obj_modifier.params
    assert yaml_modifier.layers == serialized_modifier.layers == obj_modifier.layers
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
SET_PARAM_VAL = [0 for _ in range(DEFAULT_MODEL_LAYER_PARAM_SIZE)]
SET_PARAM_MODIFIERS = [
    lambda: SetParamModifier(
        param=DEFAULT_MODEL_LAYER_PARAM,
        layers=[DEFAULT_MODEL_LAYER],
        val=SET_PARAM_VAL,
        start_epoch=0.0,
    ),
    lambda: SetParamModifier(
        param=DEFAULT_MODEL_LAYER_PARAM,
        layers=[DEFAULT_MODEL_LAYER],
        val=SET_PARAM_VAL,
        start_epoch=10.0,
    ),
]


@pytest.mark.parametrize("modifier_lambda", SET_PARAM_MODIFIERS, scope="function")
@pytest.mark.parametrize("model_lambda", [def_model], scope="function")
@pytest.mark.parametrize("optim_lambda", [def_optim_sgd], scope="function")
class TestSetParamModifierImpl(ScheduledModifierTest):
    def test_lifecycle(
        self, modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
    ):
        modifier, model, optimizer = self.create_test_objs(
            modifier_lambda, model_lambda, optim_lambda
        )
        self._lifecycle_helper(modifier, model, optimizer)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires cuda availability"
    )
    def test_lifecycle_cuda(
        self, modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
    ):
        modifier, model, optimizer = self.create_test_objs(
            modifier_lambda, model_lambda, optim_lambda
        )
        model = model.to("cuda")
        self._lifecycle_helper(modifier, model, optimizer)

    def _lifecycle_helper(self, modifier, model, optimizer):
        param = get_layer_param(DEFAULT_MODEL_LAYER_PARAM, DEFAULT_MODEL_LAYER, model)
        self.initialize_helper(modifier, model, optimizer)

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


def test_set_param_yaml():
    param_strict = False
    start_epoch = 10.0
    yaml_str = """
    !SetParamModifier
        param: {}
        layers:
            - {}
        val: {}
        param_strict: {}
        start_epoch: {}
    """.format(
        DEFAULT_MODEL_LAYER_PARAM,
        DEFAULT_MODEL_LAYER,
        SET_PARAM_VAL,
        param_strict,
        start_epoch,
    )
    yaml_modifier = SetParamModifier.load_obj(yaml_str)  # type: SetParamModifier
    serialized_modifier = SetParamModifier.load_obj(
        str(yaml_modifier)
    )  # type: SetParamModifier
    obj_modifier = SetParamModifier(
        param=DEFAULT_MODEL_LAYER_PARAM,
        layers=[DEFAULT_MODEL_LAYER],
        val=SET_PARAM_VAL,
        param_strict=param_strict,
        start_epoch=start_epoch,
    )

    assert isinstance(yaml_modifier, SetParamModifier)
    assert yaml_modifier.param == serialized_modifier.param == obj_modifier.param
    assert yaml_modifier.layers == obj_modifier.layers == obj_modifier.layers
    assert yaml_modifier.val == obj_modifier.val == obj_modifier.val
    assert (
        yaml_modifier.param_strict
        == obj_modifier.param_strict
        == obj_modifier.param_strict
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
GRADUAL_PARAM_INIT = [0.1 * (c + 1) for c in range(DEFAULT_MODEL_LAYER_PARAM_SIZE)]
GRADUAL_PARAM_FINAL = [1.0 * (c + 1) for c in range(DEFAULT_MODEL_LAYER_PARAM_SIZE)]
GRADUAL_PARAM_MODIFIERS = [
    lambda: GradualParamModifier(
        param=DEFAULT_MODEL_LAYER_PARAM,
        layers=[DEFAULT_MODEL_LAYER],
        init_val=GRADUAL_PARAM_INIT,
        final_val=GRADUAL_PARAM_FINAL,
        start_epoch=0.0,
        end_epoch=10.0,
        update_frequency=1.0,
        inter_func="linear",
        param_strict=True,
    ),
    lambda: GradualParamModifier(
        param=DEFAULT_MODEL_LAYER_PARAM,
        layers=[DEFAULT_MODEL_LAYER],
        init_val=GRADUAL_PARAM_INIT,
        final_val=GRADUAL_PARAM_FINAL,
        start_epoch=10.0,
        end_epoch=20.0,
        update_frequency=1.0,
        inter_func="linear",
        param_strict=True,
    ),
]


@pytest.mark.parametrize("modifier_lambda", GRADUAL_PARAM_MODIFIERS, scope="function")
@pytest.mark.parametrize("model_lambda", [def_model], scope="function")
@pytest.mark.parametrize("optim_lambda", [def_optim_sgd], scope="function")
class TestGradualParamModifierImpl(ScheduledUpdateModifierTest):
    def test_lifecycle(
        self, modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
    ):
        modifier, model, optimizer = self.create_test_objs(
            modifier_lambda, model_lambda, optim_lambda
        )
        self._lifecycle_helper(modifier, model, optimizer)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires cuda availability"
    )
    def test_lifecycle_cuda(
        self, modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
    ):
        modifier, model, optimizer = self.create_test_objs(
            modifier_lambda, model_lambda, optim_lambda
        )
        model = model.to("cuda")
        self._lifecycle_helper(modifier, model, optimizer)

    def _lifecycle_helper(self, modifier, model, optimizer):
        param = get_layer_param(DEFAULT_MODEL_LAYER_PARAM, DEFAULT_MODEL_LAYER, model)
        self.initialize_helper(modifier, model, optimizer)

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


def test_gradual_param_yaml():
    param_strict = False
    start_epoch = 10.0
    end_epoch = 20.0
    update_frequency = 1.0
    inter_func = "linear"
    yaml_str = """
    !GradualParamModifier
        param: {}
        layers:
            - {}
        init_val: {}
        final_val: {}
        start_epoch: {}
        end_epoch: {}
        update_frequency: {}
        inter_func: {}
        param_strict: {}
    """.format(
        DEFAULT_MODEL_LAYER_PARAM,
        DEFAULT_MODEL_LAYER,
        GRADUAL_PARAM_INIT,
        GRADUAL_PARAM_FINAL,
        start_epoch,
        end_epoch,
        update_frequency,
        inter_func,
        param_strict,
    )
    yaml_modifier = GradualParamModifier.load_obj(
        yaml_str
    )  # type: GradualParamModifier
    serialized_modifier = GradualParamModifier.load_obj(
        str(yaml_modifier)
    )  # type: GradualParamModifier
    obj_modifier = GradualParamModifier(
        param=DEFAULT_MODEL_LAYER_PARAM,
        layers=[DEFAULT_MODEL_LAYER],
        init_val=GRADUAL_PARAM_INIT,
        final_val=GRADUAL_PARAM_FINAL,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        update_frequency=update_frequency,
        inter_func=inter_func,
        param_strict=param_strict,
    )

    assert isinstance(yaml_modifier, GradualParamModifier)
    assert yaml_modifier.param == serialized_modifier.param == obj_modifier.param
    assert yaml_modifier.layers == serialized_modifier.layers == obj_modifier.layers
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
        yaml_modifier.param_strict
        == serialized_modifier.param_strict
        == obj_modifier.param_strict
    )
