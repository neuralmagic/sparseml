from typing import Union
from torch.nn import Module

from neuralmagicML.sparsity import convert_relus_to_fat


def edit_model(model: Module, model_tag: Union[None, str]):
    converted = convert_relus_to_fat(model, inplace=True)
    print('converted {} relus to FAT relus'.format(len(converted)))

    return model
