from typing import Union
from torch.nn import Module

from neuralmagicML.sparsity import convert_relus_to_fat


def handle_models(model: Module, teacher: Union[Module, None]):
    converted = convert_relus_to_fat(model, inplace=True)
    print('converted {} relus to FAT relus'.format(len(converted)))
    print(model)
