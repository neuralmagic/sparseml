from typing import Union
from torch.nn import Module

from neuralmagicML.sparsity import convert_relus_to_fat
from neuralmagicML.sparsity.activation import FATReluType


def edit_model(model: Module, model_tag: Union[None, str]):
    converted = convert_relus_to_fat(model,  FATReluType.exponential, threshold=0.0, compression=1.0)
    print('converted {} relus to FAT exponential relus'.format(len(converted)))

    return model
