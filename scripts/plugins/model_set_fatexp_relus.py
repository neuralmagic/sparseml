from typing import Union
from torch.nn import Module

from neuralmagicML.sparsity import convert_relus_to_fat
from neuralmagicML.sparsity.activation import FATReluType


def handle_models(model: Module, teacher: Union[Module, None]):
    fat_exp_relu_init_kwargs = {'threshold': 0.0, 'compression': 1.0}

    converted = convert_relus_to_fat(model,  FATReluType.exponential, **fat_exp_relu_init_kwargs)
    print('converted {} relus to FAT exponential relus'.format(len(converted)))
    print(model)
