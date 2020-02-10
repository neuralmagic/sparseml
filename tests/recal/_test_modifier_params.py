import pytest

from typing import List
from collections import OrderedDict
import sys
import yaml
import torch
from torch import Tensor
from torch.nn import Module, Sequential, Parameter, Linear
from torch.optim import SGD, Optimizer

from neuralmagicML.recal import (
    TrainableParamsModifier,
    SetParamModifier,
    GradualParamModifier,
)


@pytest.fixture
def optim() -> Optimizer:
    return SGD([Parameter(torch.tensor(0.0), requires_grad=True)], lr=INIT_LR)


@pytest.fixture
def loss() -> Tensor:
    return torch.tensor(0.0)


INIT_LR = 0.00001
SET_LR = 0.1
START_EPOCH = 1.0
DEFAULT_EPOCH = 0.0
DEFAULT_STEPS_PER_EPOCH = 10000
DEFAULT_LAYER = "block1.fc2"
DEFAULT_LAYER_BIAS_SIZE = 8


def _create_models() -> List[Module]:
    return [
        Sequential(
            OrderedDict(
                [
                    ("fc1", Linear(8, 16, bias=True)),
                    ("fc2", Linear(16, 32, bias=True)),
                    (
                        "block1",
                        Sequential(
                            OrderedDict(
                                [
                                    ("fc1", Linear(32, 16, bias=True)),
                                    ("fc2", Linear(16, 8, bias=True)),
                                ]
                            )
                        ),
                    ),
                ]
            )
        )
    ]


##############################
#
# TrainableParamsModifier tests
#
##############################


@pytest.mark.device_cpu
@pytest.mark.parametrize("model", _create_models())
def test_trainable_const_lifecycle(model, optim, loss):
    params = "__ALL__"
    layers = "__ALL__"
    modifier = TrainableParamsModifier(
        params, layers, trainable=False, start_epoch=0.0, end_epoch=10.0
    )
    modifier.initialize(model, optim)
    modifier.update(model, optim, 0.0, DEFAULT_STEPS_PER_EPOCH)

    for param in model.parameters():
        assert not param.requires_grad


@pytest.mark.device_cpu
@pytest.mark.parametrize("model", _create_models())
def test_trainable_yaml_lifecycle(model, optim, loss):
    yaml_string = """
    !TrainableParamsModifier
        params: __ALL__
        layers: __ALL__
        trainable: False
        start_epoch: 0
        end_epoch: 10
    """
    modifier = yaml.safe_load(yaml_string)
    modifier.initialize(model, optim)
    modifier.update(model, optim, 0.0, DEFAULT_STEPS_PER_EPOCH)

    for param in model.parameters():
        assert not param.requires_grad


@pytest.mark.device_cpu
@pytest.mark.parametrize("model", _create_models())
def test_trainable_yaml_write_read(model, optim, loss):
    params = "__ALL__"
    layers = "__ALL__"
    write_modifier = TrainableParamsModifier(
        params, layers, trainable=False, start_epoch=0.0, end_epoch=10.0
    )
    yaml_str = yaml.dump(write_modifier)
    read_modifier = yaml.load(yaml_str)  # type: TrainableParamsModifier
    assert write_modifier.params == read_modifier.params
    assert write_modifier.layers == read_modifier.layers
    assert write_modifier.trainable == read_modifier.trainable
    assert write_modifier.params_strict == read_modifier.params_strict
    assert write_modifier.start_epoch == read_modifier.start_epoch
    assert write_modifier.end_epoch == read_modifier.end_epoch


@pytest.mark.device_cpu
@pytest.mark.parametrize("model", _create_models())
def test_trainable_start_indefinite(model, optim, loss):
    params = "__ALL__"
    layers = "__ALL__"
    modifier = TrainableParamsModifier(params, layers, trainable=False, start_epoch=0.0)
    modifier.initialize(model, optim)

    for epoch in range(0, 20):
        if epoch == 0:
            assert modifier.update_ready(epoch, DEFAULT_STEPS_PER_EPOCH)
            modifier.scheduled_update(model, optim, epoch, DEFAULT_STEPS_PER_EPOCH)
        else:
            assert not modifier.update_ready(epoch, DEFAULT_STEPS_PER_EPOCH)

        for param in model.parameters():
            assert not param.requires_grad


@pytest.mark.device_cpu
@pytest.mark.parametrize("model", _create_models())
def test_trainable_start_end(model, optim, loss):
    params = "__ALL__"
    layers = "__ALL__"
    modifier = TrainableParamsModifier(
        params, layers, trainable=False, start_epoch=0.0, end_epoch=10.0
    )
    modifier.initialize(model, optim)

    for epoch in range(0, 20):
        if epoch == 0 or epoch == 10:
            assert modifier.update_ready(epoch, DEFAULT_STEPS_PER_EPOCH)
            modifier.scheduled_update(model, optim, epoch, DEFAULT_STEPS_PER_EPOCH)
        else:
            assert not modifier.update_ready(epoch, DEFAULT_STEPS_PER_EPOCH)

        if epoch < 10:
            for param in model.parameters():
                assert not param.requires_grad
        else:
            for param in model.parameters():
                assert param.requires_grad


@pytest.mark.device_cpu
@pytest.mark.parametrize("model", _create_models())
def test_trainable_delayed_start_indefinite(model, optim, loss):
    params = "__ALL__"
    layers = "__ALL__"
    modifier = TrainableParamsModifier(
        params, layers, trainable=False, start_epoch=10.0
    )
    modifier.initialize(model, optim)

    for epoch in range(0, 30):
        if epoch == 10:
            assert modifier.update_ready(epoch, DEFAULT_STEPS_PER_EPOCH)
            modifier.scheduled_update(model, optim, epoch, DEFAULT_STEPS_PER_EPOCH)
        else:
            assert not modifier.update_ready(epoch, DEFAULT_STEPS_PER_EPOCH)

        if epoch < 10:
            for param in model.parameters():
                assert param.requires_grad
        else:
            for param in model.parameters():
                assert not param.requires_grad


@pytest.mark.device_cpu
@pytest.mark.parametrize("model", _create_models())
def test_trainable_delayed_start_end(model, optim, loss):
    params = "__ALL__"
    layers = "__ALL__"
    modifier = TrainableParamsModifier(
        params, layers, trainable=False, start_epoch=10.0, end_epoch=20.0
    )
    modifier.initialize(model, optim)

    for epoch in range(0, 30):
        if epoch == 10 or epoch == 20:
            assert modifier.update_ready(epoch, DEFAULT_STEPS_PER_EPOCH)
            modifier.scheduled_update(model, optim, epoch, DEFAULT_STEPS_PER_EPOCH)
        else:
            assert not modifier.update_ready(epoch, DEFAULT_STEPS_PER_EPOCH)

        if epoch < 10 or epoch >= 20:
            for param in model.parameters():
                assert param.requires_grad
        else:
            for param in model.parameters():
                assert not param.requires_grad


@pytest.mark.device_cpu
@pytest.mark.parametrize("model", _create_models())
def test_trainable_specific(model, optim, loss):
    params = ["weight"]
    layers = [
        name
        for index, (name, mod) in enumerate(model.named_modules())
        if "fc" in name and index % 2 == 0
    ]
    modifier = TrainableParamsModifier(
        params, layers, trainable=False, start_epoch=0.0, end_epoch=10.0
    )
    modifier.initialize(model, optim)

    for epoch in range(0, 30):
        if epoch == 0 or epoch == 10:
            assert modifier.update_ready(epoch, DEFAULT_STEPS_PER_EPOCH)
            modifier.scheduled_update(model, optim, epoch, DEFAULT_STEPS_PER_EPOCH)
        else:
            assert not modifier.update_ready(epoch, DEFAULT_STEPS_PER_EPOCH)

        if epoch < 10:
            for name, param in model.named_parameters():
                layer_name = ".".join(name.split(".")[:-1])
                param_name = name.split(".")[-1]

                if layer_name in layers and param_name in params:
                    assert not param.requires_grad
                else:
                    assert param.requires_grad
        else:
            for param in model.parameters():
                assert param.requires_grad


##############################
#
# SetParamModifier tests
#
##############################


@pytest.mark.device_cpu
@pytest.mark.parametrize("model", _create_models())
def test_set_const_lifecycle(model, optim, loss):
    param = "bias"
    layers = [DEFAULT_LAYER]
    val = [0.1 * counter for counter in range(DEFAULT_LAYER_BIAS_SIZE)]
    modifier = SetParamModifier(param, layers, val, start_epoch=0.0)
    modifier.initialize(model, optim)
    modifier.update(model, optim, 0.0, DEFAULT_STEPS_PER_EPOCH)

    for name, par in model.named_parameters():
        if name == "{}.{}".format(layers[0], param):
            for val1, val2 in zip(val, par.data):
                assert val1 == val2


@pytest.mark.device_cpu
@pytest.mark.parametrize("model", _create_models())
def test_set_yaml_lifecycle(model, optim, loss):
    param = "bias"
    layers = [DEFAULT_LAYER]
    val = [0.1 * counter for counter in range(DEFAULT_LAYER_BIAS_SIZE)]
    yaml_string = """
        !SetParamModifier
            param: bias
            layers:
                - {}
            val: {}
            start_epoch: 0
        """.format(
        layers[0], val
    )
    modifier = yaml.safe_load(yaml_string)
    modifier.initialize(model, optim)
    modifier.update(model, optim, 0.0, DEFAULT_STEPS_PER_EPOCH)

    for name, par in model.named_parameters():
        if name == "{}.{}".format(layers[0], param):
            for val1, val2 in zip(val, par.data):
                assert val1 == val2


@pytest.mark.device_cpu
@pytest.mark.parametrize("model", _create_models())
def test_set_yaml_write_read(model, optim, loss):
    param = "bias"
    layers = [DEFAULT_LAYER]
    val = [0.1 * counter for counter in range(DEFAULT_LAYER_BIAS_SIZE)]
    write_modifier = SetParamModifier(param, layers, val, start_epoch=0.0)
    yaml_string = yaml.dump(write_modifier)
    read_modifier = yaml.load(yaml_string)  # type: SetParamModifier
    assert write_modifier.param == read_modifier.param
    assert write_modifier.layers == read_modifier.layers
    assert write_modifier.val == read_modifier.val
    assert write_modifier.param_strict == read_modifier.param_strict
    assert write_modifier.start_epoch == read_modifier.start_epoch


@pytest.mark.device_cpu
@pytest.mark.parametrize("model", _create_models())
def test_set_start(model, optim, loss):
    param = "bias"
    layers = [DEFAULT_LAYER]
    val = [0.1 * counter for counter in range(DEFAULT_LAYER_BIAS_SIZE)]
    modifier = SetParamModifier(param, layers, val, start_epoch=0.0)
    modifier.initialize(model, optim)

    for epoch in range(15):
        if epoch == 0:
            for name, par in model.named_parameters():
                if name == "{}.{}".format(layers[0], param):
                    for val1, val2 in zip(val, par.data):
                        assert val1 != val2

            assert modifier.update_ready(epoch, DEFAULT_STEPS_PER_EPOCH)
            modifier.update(model, optim, epoch, DEFAULT_STEPS_PER_EPOCH)
        else:
            assert not modifier.update_ready(epoch, DEFAULT_STEPS_PER_EPOCH)

        for name, par in model.named_parameters():
            if name == "{}.{}".format(layers[0], param):
                for val1, val2 in zip(val, par.data):
                    assert val1 == val2


@pytest.mark.device_cuda
@pytest.mark.parametrize("model", _create_models())
def test_set_start_cuda(model, optim, loss):
    assert torch.cuda.is_available()
    model = model.to("cuda")
    param = "bias"
    layers = [DEFAULT_LAYER]
    val = [0.1 * counter for counter in range(DEFAULT_LAYER_BIAS_SIZE)]
    modifier = SetParamModifier(param, layers, val, start_epoch=0.0)
    modifier.initialize(model, optim)

    for epoch in range(15):
        if epoch == 0:
            for name, par in model.named_parameters():
                if name == "{}.{}".format(layers[0], param):
                    for val1, val2 in zip(val, par.data):
                        assert val1 != val2

            assert modifier.update_ready(epoch, DEFAULT_STEPS_PER_EPOCH)
            modifier.update(model, optim, epoch, DEFAULT_STEPS_PER_EPOCH)
        else:
            assert not modifier.update_ready(epoch, DEFAULT_STEPS_PER_EPOCH)

        for name, par in model.named_parameters():
            if name == "{}.{}".format(layers[0], param):
                for val1, val2 in zip(val, par.data):
                    assert val1 == val2


@pytest.mark.device_cpu
@pytest.mark.parametrize("model", _create_models())
def test_set_delayed_start(model, optim, loss):
    param = "bias"
    layers = [DEFAULT_LAYER]
    val = [0.1 * counter for counter in range(DEFAULT_LAYER_BIAS_SIZE)]
    modifier = SetParamModifier(param, layers, val, start_epoch=10.0)
    modifier.initialize(model, optim)

    for epoch in range(15):
        if epoch < 16:
            for name, par in model.named_parameters():
                if name == "{}.{}".format(layers[0], param):
                    for val1, val2 in zip(val, par.data):
                        assert val1 != val2

        if epoch == 15:
            assert modifier.update_ready(epoch, DEFAULT_STEPS_PER_EPOCH)
            modifier.update(model, optim, epoch, DEFAULT_STEPS_PER_EPOCH)
        else:
            assert not modifier.update_ready(epoch, DEFAULT_STEPS_PER_EPOCH)

        if epoch > 14:
            for name, par in model.named_parameters():
                if name == "{}.{}".format(layers[0], param):
                    for val1, val2 in zip(val, par.data):
                        assert val1 == val2


##############################
#
# GradualParamModifier tests
#
##############################


@pytest.mark.device_cpu
@pytest.mark.parametrize("model", _create_models())
def test_gradual_const_lifecycle(model, optim, loss):
    param = "bias"
    layers = [DEFAULT_LAYER]
    init_val = [0.1 * counter for counter in range(DEFAULT_LAYER_BIAS_SIZE)]
    final_val = [1.0 * counter for counter in range(DEFAULT_LAYER_BIAS_SIZE)]
    modifier = GradualParamModifier(
        param,
        layers,
        init_val,
        final_val,
        start_epoch=0.0,
        end_epoch=10.0,
        update_frequency=1.0,
    )
    modifier.initialize(model, optim)
    modifier.update(model, optim, 0.0, DEFAULT_STEPS_PER_EPOCH)

    for name, par in model.named_parameters():
        if name == "{}.{}".format(layers[0], param):
            for val1, val2 in zip(init_val, par.data):
                assert val1 == val2


@pytest.mark.device_cpu
@pytest.mark.parametrize("model", _create_models())
def test_gradual_yaml_lifecycle(model, optim, loss):
    param = "bias"
    layers = [DEFAULT_LAYER]
    init_val = [0.1 * counter for counter in range(DEFAULT_LAYER_BIAS_SIZE)]
    final_val = [1.0 * counter for counter in range(DEFAULT_LAYER_BIAS_SIZE)]
    yaml_string = """
        !GradualParamModifier
            param: {}
            layers:
                - {}
            init_val: {}
            final_val: {}
            start_epoch: 0
            end_epoch: 10.0
            update_frequency: 1.0
        """.format(
        param, layers[0], init_val, final_val
    )
    modifier = yaml.safe_load(yaml_string)
    modifier.initialize(model, optim)
    modifier.update(model, optim, 0.0, DEFAULT_STEPS_PER_EPOCH)

    for name, par in model.named_parameters():
        if name == "{}.{}".format(layers[0], param):
            for val1, val2 in zip(init_val, par.data):
                assert val1 == val2


@pytest.mark.device_cpu
@pytest.mark.parametrize("model", _create_models())
def test_gradual_yaml_write_read(model, optim, loss):
    param = "bias"
    layers = [DEFAULT_LAYER]
    init_val = [0.1 * counter for counter in range(DEFAULT_LAYER_BIAS_SIZE)]
    final_val = [1.0 * counter for counter in range(DEFAULT_LAYER_BIAS_SIZE)]
    write_modifier = GradualParamModifier(
        param,
        layers,
        init_val,
        final_val,
        start_epoch=0.0,
        end_epoch=10.0,
        update_frequency=1.0,
    )
    yaml_string = yaml.dump(write_modifier)
    read_modifier = yaml.load(yaml_string)  # type: GradualParamModifier
    assert write_modifier.param == read_modifier.param
    assert write_modifier.layers == read_modifier.layers
    assert write_modifier.init_val == read_modifier.init_val
    assert write_modifier.final_val == read_modifier.final_val
    assert write_modifier.param_strict == read_modifier.param_strict
    assert write_modifier.inter_func == read_modifier.inter_func
    assert write_modifier.start_epoch == read_modifier.start_epoch
    assert write_modifier.end_epoch == read_modifier.end_epoch
    assert write_modifier.update_frequency == read_modifier.update_frequency


@pytest.mark.device_cpu
@pytest.mark.parametrize("model", _create_models())
def test_gradual_start(model, optim, loss):
    param = "bias"
    layers = [DEFAULT_LAYER]
    init_val = [0.1 * counter for counter in range(DEFAULT_LAYER_BIAS_SIZE)]
    final_val = [1.0 * counter + 1.0 for counter in range(DEFAULT_LAYER_BIAS_SIZE)]
    modifier = GradualParamModifier(
        param,
        layers,
        init_val,
        final_val,
        start_epoch=0.0,
        end_epoch=10.0,
        update_frequency=1.0,
    )
    modifier.initialize(model, optim)

    for name, par in model.named_parameters():
        if name == "{}.{}".format(layers[0], param):
            for val1, val2 in zip(init_val, par.data):
                assert val1 != val2

    vals = [val - 0.0001 for val in init_val]

    for epoch in range(15):
        if epoch < 10:
            assert modifier.update_ready(epoch, DEFAULT_STEPS_PER_EPOCH)
            modifier.update(model, optim, epoch, DEFAULT_STEPS_PER_EPOCH)

            for name, par in model.named_parameters():
                if name == "{}.{}".format(layers[0], param):
                    for index, (val1, val2) in enumerate(zip(vals, par.data)):
                        assert val1 < val2
                        vals[index] = float(val2)
        else:
            if epoch == 10:
                assert modifier.update_ready(epoch, DEFAULT_STEPS_PER_EPOCH)
                modifier.update(model, optim, epoch, DEFAULT_STEPS_PER_EPOCH)

            for name, par in model.named_parameters():
                if name == "{}.{}".format(layers[0], param):
                    for val1, val2 in zip(final_val, par.data):
                        assert val1 == val2


@pytest.mark.device_cpu
@pytest.mark.parametrize("model", _create_models())
def test_gradual_delayed_start(model, optim, loss):
    param = "bias"
    layers = [DEFAULT_LAYER]
    init_val = [0.1 * counter for counter in range(DEFAULT_LAYER_BIAS_SIZE)]
    final_val = [1.0 * counter + 1.0 for counter in range(DEFAULT_LAYER_BIAS_SIZE)]
    modifier = GradualParamModifier(
        param,
        layers,
        init_val,
        final_val,
        start_epoch=10.0,
        end_epoch=20.0,
        update_frequency=1.0,
    )
    modifier.initialize(model, optim)

    vals = [val - 0.0001 for val in init_val]

    for epoch in range(30):
        if epoch < 10:
            assert not modifier.update_ready(epoch, DEFAULT_STEPS_PER_EPOCH)

            for name, par in model.named_parameters():
                if name == "{}.{}".format(layers[0], param):
                    for index, (val1, val2) in enumerate(zip(init_val, par.data)):
                        assert val1 != val2
        elif epoch < 20:
            assert modifier.update_ready(epoch, DEFAULT_STEPS_PER_EPOCH)
            modifier.update(model, optim, epoch, DEFAULT_STEPS_PER_EPOCH)

            for name, par in model.named_parameters():
                if name == "{}.{}".format(layers[0], param):
                    for index, (val1, val2) in enumerate(zip(vals, par.data)):
                        assert val1 < val2
                        vals[index] = float(val2)
        else:
            if epoch == 20:
                assert modifier.update_ready(epoch, DEFAULT_STEPS_PER_EPOCH)
                modifier.update(model, optim, epoch, DEFAULT_STEPS_PER_EPOCH)

            for name, par in model.named_parameters():
                if name == "{}.{}".format(layers[0], param):
                    for val1, val2 in zip(final_val, par.data):
                        assert val1 == val2


# @pytest.mark.device_cuda
# @pytest.mark.parametrize("model", _create_models())
# def test_gradual_start_cuda(model, optim, loss):
#     assert torch.cuda.is_available()
#     model = model.to('cuda')
#     param = 'bias'
#     layers = [DEFAULT_LAYER]
#     val = [0.1 * counter for counter in range(DEFAULT_LAYER_BIAS_SIZE)]
#     modifier = SetParamModifier(
#         param, layers, val, start_epoch=0.0
#     )
#     modifier.initialize(model, optim)
#
#     for name, par in model.named_parameters():
#         if name == '{}.{}'.format(layers[0], param):
#             for val1, val2 in zip(val, par.data):
#                 assert val1 != val2
#
#     for epoch in range(15):
#         modifier.update(model, optim, epoch, DEFAULT_STEPS_PER_EPOCH)
#
#         for name, par in model.named_parameters():
#             if name == '{}.{}'.format(layers[0], param):
#                 for val1, val2 in zip(val, par.data):
#                     assert val1 == val2.cpu()
