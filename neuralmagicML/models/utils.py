from typing import Dict, Union, List, Tuple, Callable
import os
import hashlib
import errno
from urllib.request import urlopen
import shutil
import tempfile
from tqdm import tqdm
import torch
from torch import Tensor
from torch.nn import DataParallel, Module
from torch.optim.optimizer import Optimizer
from collections import OrderedDict


__all__ = ['copy_state_dict_value', 'copy_state_dict_linear', 'copy_state_dict_conv', 'copy_state_dict_batch_norm',
           'load_pretrained_model', 'load_model', 'load_optimizer', 'save_model',
           'create_model', 'model_to_device', 'parallelize_model', 'device_to_name_ids']


def copy_state_dict_value(target_key: str, source_key: str, target: Dict[str, Tensor], source: Dict[str, Tensor],
                          delete_from_source: bool = False):
    if source_key not in source:
        raise KeyError('{} not found in source dict'.format(source_key))

    target[target_key] = source[source_key]

    if delete_from_source:
        del source[source_key]


def copy_state_dict_linear(target_name: str, source_name: str, target: Dict[str, Tensor], source: Dict[str, Tensor],
                           bias: bool = True, delete_from_source: bool = False):
    copy_state_dict_value('{}.weight'.format(target_name), '{}.weight'.format(source_name),
                          target, source, delete_from_source)

    if bias:
        copy_state_dict_value('{}.bias'.format(target_name), '{}.bias'.format(source_name),
                              target, source, delete_from_source)


def copy_state_dict_conv(target_name: str, source_name: str, target: Dict[str, Tensor], source: Dict[str, Tensor],
                         bias: bool = True, delete_from_source: bool = False):
    copy_state_dict_value('{}.weight'.format(target_name), '{}.weight'.format(source_name),
                          target, source, delete_from_source)

    if bias:
        copy_state_dict_value('{}.bias'.format(target_name), '{}.bias'.format(source_name),
                              target, source, delete_from_source)


def copy_state_dict_batch_norm(target_name: str, source_name: str, target: Dict[str, Tensor], source: Dict[str, Tensor],
                               delete_from_source: bool = False):
    copy_state_dict_value('{}.weight'.format(target_name), '{}.weight'.format(source_name),
                          target, source, delete_from_source)
    copy_state_dict_value('{}.bias'.format(target_name), '{}.bias'.format(source_name),
                          target, source, delete_from_source)
    copy_state_dict_value('{}.running_mean'.format(target_name), '{}.running_mean'.format(source_name),
                          target, source, delete_from_source)
    copy_state_dict_value('{}.running_var'.format(target_name), '{}.running_var'.format(source_name),
                          target, source, delete_from_source)

    if delete_from_source and '{}.num_batches_tracked'.format(source_name) in source:
        del source['{}.num_batches_tracked'.format(source_name)]


def load_pretrained_model(model: Module, pretrained_key: str, model_arch: str,
                          model_domain: str = 'vision', default_pretrained_key: str = 'imagenet/dense',
                          ignore_tensors: Union[None, List[str]] = None):
    if not pretrained_key:
        pretrained_key = default_pretrained_key

    url = 'https://storage.googleapis.com/models.neuralmagic.com/{}/{}/{}.pth'.format(model_domain, model_arch,
                                                                                      pretrained_key)

    cache_dir = os.path.join(os.path.abspath(os.path.expanduser('~')), '.cache', 'nm_models')
    cache_name = hashlib.md5(url.encode('utf-8')).hexdigest()

    try:
        os.makedirs(cache_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    cache_file = os.path.join(cache_dir, cache_name)

    if not os.path.exists(cache_file):
        print('downloading {} for {}'.format(model_arch, pretrained_key))
        connection = urlopen(url)
        meta = connection.info()
        content_length = meta.getheaders('Content-Length') if hasattr(meta, 'getheaders') \
            else meta.get_all('Content-Length')
        file_size = int(content_length[0]) if content_length is not None and len(content_length) > 0 else None
        temp = tempfile.NamedTemporaryFile(delete=False)

        try:
            with tqdm(total=file_size) as progress:
                while True:
                    buffer = connection.read(8192)
                    if len(buffer) == 0:
                        break
                    temp.write(buffer)
                    progress.update(len(buffer))

            temp.close()
            shutil.move(temp.name, cache_file)
        finally:
            temp.close()

            if os.path.exists(temp.name):
                os.remove(temp.name)

    model_dict = torch.load(cache_file, map_location='cpu')['state_dict']
    to_state_dict = model.state_dict()

    if ignore_tensors is not None:
        for ignore in ignore_tensors:
            # copy over the 'to state dict' values to our current state dict for any tensor we were supposed to ignore
            # only do this though if we can't match the shape and the key exists in 'to state dict'
            if ignore not in to_state_dict:
                del model_dict[ignore]
            elif to_state_dict[ignore].shape != model_dict[ignore].shape:
                model_dict[ignore] = to_state_dict[ignore]

    model.load_state_dict(model_dict, strict=ignore_tensors is None)


def load_model(path: str, model: Module, strict: bool = True):
    model_dict = torch.load(path, map_location='cpu')
    model.load_state_dict(model_dict['state_dict'], strict)


def load_optimizer(path: str, optimizer: Optimizer) -> Union[int, None]:
    model_dict = torch.load(path, map_location='cpu')
    optimizer.load_state_dict(model_dict['optimizer'])

    if 'epoch' in model_dict:
        return model_dict['epoch']

    return None


def save_model(path: str, model: Module, optimizer: Optimizer = None, epoch: Union[int, None] = None):
    if isinstance(model, DataParallel):
        model = model.module

    save_dict = {
        'state_dict': OrderedDict()
    }

    # make sure we have the model state_dict on cpu
    for key, state in model.state_dict().items():
        copy = torch.zeros(state.shape)
        copy.copy_(state)
        save_dict['state_dict'][key] = copy

    if optimizer:
        save_dict['optimizer'] = optimizer.state_dict()

    if epoch:
        save_dict['epoch'] = epoch

    torch.save(save_dict, path)


class _DataParallel(DataParallel):
    def __getattr__(self, item):
        module = super().__getattr__('module')

        if item == 'module':
            return module

        return getattr(module, item)


def parallelize_model(model: Module, ids: Union[None, List[int]]) -> Module:
    return _DataParallel(model, ids)


MODEL_MAPPINGS = {}  # type: Dict[str, Callable]


def create_model(model_type: str, model_path: Union[None, str] = None, load_strict: bool = True, **kwargs) -> Module:
    if model_type == 'help':
        raise Exception('model_type given of help, available models: \n{}'.format(list(MODEL_MAPPINGS.keys())))

    if model_type not in MODEL_MAPPINGS:
        raise ValueError('Unsupported model_type given of {}, available models: \n{}'
                         .format(model_type, list(MODEL_MAPPINGS.keys())))

    constructor = MODEL_MAPPINGS[model_type]
    model = constructor(**kwargs)  # type: Module

    if model_path:
        load_model(model_path, model, strict=load_strict)

    return model


def model_to_device(model: Module, device: str) -> Tuple[Module, str, Union[None, List[int]]]:
    device, ids = device_to_name_ids(device)

    if ids is not None:
        model = parallelize_model(model, ids)

    model = model.to(device)

    return model, device, ids


def device_to_name_ids(device: str) -> Tuple[str, Union[None, List[int]]]:
    split = device.split(':')
    name = split[0]

    if name == 'cpu':
        return name, None

    if name != 'cuda' or not torch.cuda.is_available():
        raise ValueError('{} device not available on this system'.format(name))

    if len(split) < 2:
        return name, None

    ids = [int(id_) for id_ in split[1].split(',')]
    count = torch.cuda.device_count()

    for id_ in ids:
        if id_ >= count:
            raise ValueError('{} device id not available on this system'.format(id_))

    if len(ids) == 1:
        return '{}:{}'.format(name, ids[0]), None

    return name, ids
