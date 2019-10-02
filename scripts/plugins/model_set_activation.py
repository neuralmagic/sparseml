from typing import Dict
from torch.nn import Module

from neuralmagicML.nn import replace_activation, is_activation


def edit_model(model: Module, state_loaded: bool, **kwargs):
    """
    plugin to convert activations in a model to a given activation type
    called twice, one for after module initialization and again after loading weights

    accepted kwargs:
      act_type: the activation type to replace current activations with, see nn.activations for full list
                ex: relu, relu6, prelu, lrelu
                defaults to relu
      convert_after: determine when to convert the ReLUs into PReLUs
                     either 'init' to convert after module initialization or
                     'load' to convert after loading module state
                     defaults to init

    :param model: the model to edit and change out ReLUs for FATReLUs
    :param state_loaded: True if model state_dict has been loaded, False otherwise
    :param kwargs: additional args that can be used to describe how to convert, see accepted kwargs above
    :return: the edited model
    """
    act_type = 'relu' if 'act_type' not in kwargs else kwargs['act_type']
    convert_after = 'init' if 'convert_after' not in kwargs else kwargs['convert_after']

    if (convert_after == 'init' and not state_loaded) or (convert_after == 'load' and state_loaded):
        converted = _convert(model, act_type)
        print('model_set_activation plugin converted {} activations to {}'.format(len(converted), act_type))
        print(model)

    return model


def _convert(model: Module, act_type: str) -> Dict[str, Module]:
    act_keys = []

    for name, mod in model.named_modules():
        if is_activation(mod):
            act_keys.append(name)

    added = {}

    for key in act_keys:
        added[key] = replace_activation(model, key, act_type, inplace=True)

    return added
