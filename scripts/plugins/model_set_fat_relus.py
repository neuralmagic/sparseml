from torch.nn import Module

from neuralmagicML.sparsity import convert_relus_to_fat


def edit_model(model: Module, state_loaded: bool, **kwargs):
    """
    plugin to convert ReLUs in a model to FATReLUs
    called twice, one for after module initialization and again after loading weights

    accepted kwargs:
      fat_convert_after: determine when to convert the ReLUs into FATReLUs
                         either 'init' to convert after module initialization or
                         'load' to convert after loading module state
                         defaults to init

    :param model: the model to edit and change out ReLUs for FATReLUs
    :param state_loaded: True if model state_dict has been loaded, False otherwise
    :param kwargs: additional args that can be used to describe how to convert, see accepted kwargs above
    :return: the edited model
    """
    fat_convert_after = 'init' if 'fat_convert_after' not in kwargs else kwargs['fat_convert_after']

    if (fat_convert_after == 'init' and not state_loaded) or (fat_convert_after == 'load' and state_loaded):
        converted = convert_relus_to_fat(model, inplace=True)
        print('model_set_fat_relus plugin converted {} relus to FAT relus'.format(len(converted)))

    return model
