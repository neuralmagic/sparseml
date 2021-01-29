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

"""
Utility / helper functions
"""

import random
import re
from collections import OrderedDict, namedtuple
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy
import torch
from torch import Tensor
from torch.nn import Linear, Module, Parameter
from torch.nn.modules.conv import _ConvNd
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from sparseml.utils import create_dirs, save_numpy


__all__ = [
    "default_device",
    "get_optim_learning_rate",
    "set_optim_learning_rate",
    "early_stop_data_loader",
    "infinite_data_loader",
    "tensors_batch_size",
    "tensors_to_device",
    "tensors_to_precision",
    "tensors_module_forward",
    "tensor_export",
    "tensors_export",
    "tensor_density",
    "tensor_sparsity",
    "tensor_sample",
    "mask_difference",
    "get_layer",
    "get_terminal_layers",
    "get_conv_layers",
    "get_linear_layers",
    "get_prunable_layers",
    "get_named_layers_and_params_by_regex",
    "any_str_or_regex_matches_param_name",
    "NamedLayerParam",
    "get_layer_param",
    "set_deterministic_seeds",
    "torch_distributed_zero_first",
]


##############################
#
# pytorch device helpers
#
##############################


def default_device() -> str:
    """
    :return: the device that should be defaulted to for the current setup.
        if multiple gpus are available then will return a string with all of them,
        else if single gpu available then will return cuda,
        else returns cpu
    """

    if not torch.cuda.is_available():
        return "cpu"

    if torch.cuda.device_count() < 2:
        return "cuda"

    device_ids = [str(i) for i in range(torch.cuda.device_count())]

    return "cuda:{}".format(",".join(device_ids))


##############################
#
# pytorch optim helpers
#
##############################


def get_optim_learning_rate(optim: Optimizer) -> float:
    """
    :param optim: The optimizer to get the learning rate for

    :return: convenience function to get the first learning rate for any of
        the param groups in the optimizer
    """
    for param_group in optim.param_groups:
        return param_group["lr"]

    raise RuntimeError("cannot get learning_rate, no param_groups available")


def set_optim_learning_rate(optim: Optimizer, value: float):
    """
    :param optim: The optimizer to set the learning rate for
    :param value: the learning rate to set for the optimizer,
        will set all param groups in the optim to this value
    """
    for param_group in optim.param_groups:
        param_group["lr"] = value


##############################
#
# pytorch data loader helpers
#
##############################


def early_stop_data_loader(data_loader: DataLoader, early_stop_steps: int):
    """
    An iterator that goes through the data_loader for yields and stops
    after early_stop_steps instead of the full loader

    :param data_loader: the data loader to continually repeat
    :param early_stop_steps: if set, the number of steps to run and break out early
        instead of running all of the steps in the data loader,
        if < 1 then will run the full length
    :return: an iterable for the never ending data loader
    """
    counter = 0

    for data in data_loader:
        yield data
        counter += 1

        if 0 < early_stop_steps <= counter:
            break


def infinite_data_loader(
    data_loader: DataLoader, early_stop_steps: int = -1, cache: bool = False
):
    """
    A never ending data loader that will keep repeating the one passed in.
    Will additionally cache the data if requested.

    :param data_loader: the data loader to continually repeat
    :param early_stop_steps: if set, the number of steps to run and break out early
        instead of running all of the steps in the data loader
    :param cache: True to cache the results in memory and return those on
        subsequent requests, False otherwise
    :return: an iterable for the never ending data loader
    """
    cached = None

    while True:
        if not cache or cached is None:
            cached = []

            for data in early_stop_data_loader(data_loader, early_stop_steps):
                if cache:
                    cached.append(deepcopy(data))

                yield data
        else:
            for data in cached:
                yield data


##############################
#
# pytorch tensor helper functions
#
##############################


NamedLayerParam = namedtuple(
    "NamedLayerParam", ["layer_name", "layer", "param_name", "param"]
)


def tensors_batch_size(tensors: Union[Tensor, Iterable[Tensor], Dict[Any, Tensor]]):
    """
    Default function for getting the batch size from a tensor or collection of tensors.
    Returns the batch size (zeroth index for shape) of the first found tensor.

    Supported use cases:
        - single tensor
        - Dictionary of single tensors
        - Dictionary of iterable of tensors
        - Dictionary of dictionary of tensors
        - Iterable of single tensors
        - Iterable of iterable of tensors
        - Iterable of dictionary of tensors

    :param tensors: the tensor or collection of tensors to get a batch size from,
        taken from the first found tensor
    :return: the batch size (0th element of shape) of the first contained
        tensor in the data
    """
    if isinstance(tensors, Tensor):
        return tensors.shape[0]

    if isinstance(tensors, Dict):
        for key, tens in tensors.items():
            batch_size = tensors_batch_size(tens)

            if batch_size > -1:
                return batch_size

    if isinstance(tensors, Iterable):
        for tens in tensors:
            batch_size = tensors_batch_size(tens)

            if batch_size > -1:
                return batch_size

    return -1


def tensors_to_device(
    tensors: Union[Tensor, Iterable[Tensor], Dict[Any, Tensor]], device: str
) -> Union[Tensor, Iterable[Tensor], Dict[Any, Tensor]]:
    """
    Default function for putting a tensor or collection of tensors to the proper device.
    Returns the tensor references after being placed on the proper device.

    Supported use cases:
        - single tensor
        - Dictionary of single tensors
        - Dictionary of iterable of tensors
        - Dictionary of dictionary of tensors
        - Iterable of single tensors
        - Iterable of iterable of tensors
        - Iterable of dictionary of tensors

    :param tensors: the tensors or collection of tensors to put onto a device
    :param device: the string representing the device to put the tensors on,
        ex: 'cpu', 'cuda', 'cuda:1'
    :return: the tensors or collection of tensors after being placed on the device
    """
    if isinstance(tensors, Tensor):
        return tensors.to(device)

    if isinstance(tensors, Dict):
        return {key: tensors_to_device(tens, device) for key, tens in tensors.items()}

    if isinstance(tensors, tuple):
        return tuple(tensors_to_device(tens, device) for tens in tensors)

    if isinstance(tensors, Iterable):
        return [tensors_to_device(tens, device) for tens in tensors]

    raise ValueError(
        "unrecognized type for tensors given of {}".format(tensors.__class__.__name__)
    )


def tensors_to_precision(
    tensors: Union[Tensor, Iterable[Tensor], Dict[Any, Tensor]], full_precision: bool
) -> Union[Tensor, Iterable[Tensor], Dict[Any, Tensor]]:
    """
    :param tensors: the tensors to change the precision of
    :param full_precision: True for full precision (float 32) and
        False for half (float 16)
    :return: the tensors converted to the desired precision
    """
    if isinstance(tensors, Tensor):
        return tensors.float() if full_precision else tensors.half()

    if isinstance(tensors, Dict):
        return {
            key: tensors_to_precision(tens, full_precision)
            for key, tens in tensors.items()
        }

    if isinstance(tensors, tuple):
        return tuple(tensors_to_precision(tens, full_precision) for tens in tensors)

    if isinstance(tensors, Iterable):
        return [tensors_to_precision(tens, full_precision) for tens in tensors]

    raise ValueError(
        "unrecognized type for tensors given of {}".format(tensors.__class__.__name__)
    )


def tensors_module_forward(
    tensors: Union[Tensor, Iterable[Tensor], Dict[Any, Tensor]],
    module: Module,
    check_feat_lab_inp: bool = True,
) -> Any:
    """
    Default function for calling into a model with data for a forward execution.
    Returns the model result.
    Note, if an iterable the features to be passed into the model are considered
    to be at index 0 and other indices are for labels.

    Supported use cases: single tensor,
    iterable with first tensor taken as the features to pass into the model

    :param tensors: the data to be passed into the model, if an iterable the features
        to be passed into the model are considered to be at index 0 and other indices
        are for labels
    :param module: the module to pass the data into
    :param check_feat_lab_inp: True to check if the incoming tensors looks like
        it's made up of features and labels ie a tuple or list with 2 items
        (typical output from a data loader) and will call into the model with just
        the first element assuming it's the features False to not check
    :return: the result of calling into the model for a forward pass
    """
    if (
        (isinstance(tensors, tuple) or isinstance(tensors, List))
        and len(tensors) == 2
        and check_feat_lab_inp
    ):
        # assume if this is a list or tuple of 2 items that it is made up of
        # (features, labels) pass the features into a recursive call for the model
        return tensors_module_forward(tensors[0], module, check_feat_lab_inp=False)

    if isinstance(tensors, Tensor):
        return module(tensors)

    if isinstance(tensors, Dict):
        return module(**tensors)

    if isinstance(tensors, Iterable):
        return module(*tensors)

    raise ValueError(
        "unrecognized type for data given of {}".format(tensors.__class__.__name__)
    )


def tensor_export(
    tensor: Union[Tensor, Dict[str, Tensor], Iterable[Tensor]],
    export_dir: str,
    name: str,
    npz: bool = True,
) -> str:
    """
    :param tensor: tensor to export to a saved numpy array file
    :param export_dir: the directory to export the file in
    :param name: the name of the file, .npy will be appended to it
    :param npz: True to export as an npz file, False otherwise
    :return: the path of the numpy file the tensor was exported to
    """

    if isinstance(tensor, Tensor):
        tensor = tensor.detach().cpu().numpy()
    elif isinstance(tensor, Dict):
        tensor = OrderedDict(
            (key, val.detach().cpu().numpy()) for key, val in tensor.items()
        )
    elif isinstance(tensor, Iterable):
        tensor = [val.detach().cpu().numpy() for val in tensor]
    else:
        raise ValueError("Unrecognized type given for tensorr {}".format(tensor))

    return save_numpy(tensor, export_dir, name, npz)


def tensors_export(
    tensors: Union[Tensor, Iterable[Tensor]],
    export_dir: str,
    name_prefix: str,
    counter: int = 0,
    break_batch: bool = False,
) -> List[str]:
    """
    :param tensors: the tensors to export to a saved numpy array file
    :param export_dir: the directory to export the files in
    :param name_prefix: the prefix name for the tensors to save as, will append
        info about the position of the tensor in a list or dict in addition
        to the .npy file format
    :param counter: the current counter to save the tensor at
    :param break_batch: treat the tensor as a batch and break apart into
        multiple tensors
    :return: the exported paths
    """
    create_dirs(export_dir)
    exported_paths = []
    if break_batch:
        _tensors_export_batch(tensors, export_dir, name_prefix, counter, exported_paths)
    else:
        _tensors_export_recursive(
            tensors, export_dir, name_prefix, counter, exported_paths
        )

    return exported_paths


def _tensors_export_recursive(
    tensors: Union[Tensor, Iterable[Tensor]],
    export_dir: str,
    name_prefix: str,
    counter: int,
    exported_paths: List[str],
):
    if isinstance(tensors, Tensor):
        exported_paths.append(
            tensor_export(tensors, export_dir, "{}-{:04d}".format(name_prefix, counter))
        )

        return

    if isinstance(tensors, Iterable):
        for index, tens in enumerate(tensors):
            _tensors_export_recursive(
                tens,
                export_dir,
                name_prefix,
                counter + index,
                exported_paths,
            )

        return

    raise ValueError(
        "unrecognized type for tensors given of {}".format(tensors.__class__.__name__)
    )


def _tensors_export_batch(
    tensors: Union[Tensor, Iterable[Tensor]],
    export_dir: str,
    name_prefix: str,
    counter: int,
    exported_paths: List[str],
):
    if isinstance(tensors, Tensor):
        if len(tensors.shape) == 1:
            exported_paths.append(
                tensor_export(
                    tensors, export_dir, "{}-{:04d}".format(name_prefix, counter)
                )
            )
            return

        for index, tens in enumerate(tensors):
            exported_paths.append(
                tensor_export(
                    tens, export_dir, "{}-{:04d}".format(name_prefix, counter + index)
                )
            )

        return

    if isinstance(tensors, Iterable):
        for index, tens in enumerate(zip(*tensors)):
            exported_paths.append(
                tensor_export(
                    tens, export_dir, "{}-{:04d}".format(name_prefix, counter + index)
                )
            )

        return

    raise ValueError(
        "unrecognized type for tensors given of {}".format(tensors.__class__.__name__)
    )


def tensor_sparsity(
    tens: Tensor, dim: Union[None, int, List[int], Tuple[int, ...]] = None
) -> Tensor:
    """
    :param tens: the tensor to calculate the sparsity for
    :param dim: the dimension(s) to split the calculations over;
        ex, can split over batch, channels, or combos
    :return: the sparsity of the input tens, ie the fraction of numbers that are zero
    """
    if dim is None:
        zeros = (tens == 0).sum()
        total = tens.numel()

        return zeros.float() / float(total)

    if isinstance(dim, int):
        dim = [dim]

    if max(dim) >= len(tens.shape):
        raise ValueError(
            "Unsupported dim given of {} in {} for tensor shape {}".format(
                max(dim), dim, tens.shape
            )
        )

    sum_dims = [ind for ind in range(len(tens.shape)) if ind not in dim]
    zeros = (tens == 0).sum(dim=sum_dims) if sum_dims else tens == 0
    total = numpy.prod(
        [tens.shape[ind] for ind in range(len(tens.shape)) if ind not in dim]
    )

    permute_order = sorted(
        ((d, len(dim) - i - 1) for i, d in enumerate(dim)), reverse=True
    )
    permute = [d[1] for d in permute_order]

    if permute != [i for i in range(len(permute))]:
        # need to permute to get desired dimensions at the front
        zeros = zeros.permute(*permute).contiguous()

    return zeros.float() / float(total)


def tensor_density(tens: Tensor, dim: Union[None, int, Iterable[int]] = None) -> Tensor:
    """
    :param tens: the tensor to calculate the density for
    :param dim: the dimension(s) to split the calculations over; ex, can split over
        batch, channels, or combos
    :return: the density of the input tens, ie the fraction of numbers that are non zero
    """
    density = (tensor_sparsity(tens, dim) - 1.0) * -1.0

    return density


def tensor_sample(
    tens: Tensor,
    sample_size: int,
    dim: Union[None, int, List[int], Tuple[int, ...]] = None,
) -> Tensor:
    """
    :param tens: the tensor to grab samples from
    :param sample_size: the number of samples to grab overall if dim is not supplied
        or per each dim if it is
    :param dim: the dimension(s) to split the samples over;
        ex, can split over batch, channels, or combos
    :return: the sampled tensor
    """
    if sample_size < 1:
        raise ValueError("improper sample size given of {}".format(sample_size))

    if dim is None:
        indices = tens.new_zeros((sample_size,)).long().random_(0, tens.numel())
        samples = tens.view(-1)[indices]

        return samples

    if isinstance(dim, int):
        dim = [dim]

    if max(dim) >= len(tens.shape):
        raise ValueError(
            "Unsupported dim given of {} in {} for tensor shape {}".format(
                max(dim), dim, tens.shape
            )
        )

    if dim != [ind for ind in range(len(dim))]:
        # put the desired dimension(s) at the front to sample from
        tens = tens.permute(
            *dim, *[ind for ind in range(len(tens.shape)) if ind not in dim]
        )
        dim = [ind for ind in range(len(dim))]

    if not tens.is_contiguous():
        tens = tens.contiguous()

    num_indices = int(numpy.prod([tens.shape[ind] for ind in range(len(dim))]))
    elem_per_ind = int(
        numpy.prod([tens.shape[ind] for ind in range(len(dim), len(tens.shape))])
    )
    # create a new tensor with offsets set for each of our elements that we are indexing
    indices = tens.new_tensor(
        [ind * elem_per_ind for ind in range(num_indices)], dtype=torch.long
    ).unsqueeze(1)
    # now broadcast it across to the total number of elements we should end with
    indices = indices * tens.new_ones((num_indices, sample_size), dtype=torch.long)
    # finally add in a random number within the available range per index
    indices += tens.new_zeros((num_indices, sample_size), dtype=torch.long).random_(
        0, elem_per_ind
    )
    # get our samples
    samples = tens.view(-1)[indices.view(-1)]
    # reshape for the proper dimension
    samples = samples.view(*(tens.shape[ind] for ind in dim), sample_size)

    return samples


def mask_difference(old_mask: Tensor, new_mask: Tensor) -> Tensor:
    """
    :param old_mask: the old mask to compare against for calculating the difference
    :param new_mask: the new mask to compare with for calculating the difference
    :return: a tensor representing the change from the old_mask to the new_mask
             specifically values returned as 1.0 are newly unmasked (0.0 => 1.0)
             values returned as -1.0 are newly masked (1.0 => 0.0)
             values returned as 0.0 had no change in (0.0 => 0.0 or 1.0 => 1.0)
    """
    newly_masked = ((old_mask != new_mask) & (new_mask == 0.0)).type(old_mask.type())
    newly_unmasked = ((old_mask != new_mask) & (new_mask == 1.0)).type(old_mask.type())

    return -1.0 * newly_masked + newly_unmasked


##############################
#
# pytorch module helper functions
#
##############################


def get_layer(name: str, module: Module) -> Module:
    """
    :param name: the name of the layer to grab from the module
    :param module: the module containing the layer to grab
    :return: the module representing the layer in the module
    """
    layers = name.split(".")
    layer = module

    for name in layers:
        layer = layer.__getattr__(name)

    return layer


def get_terminal_layers(module: Module) -> Dict[str, Module]:
    """
    :param module: the module to grab all terminal layers for
    :return: a list of all of the terminal layers in a model
        (ie not containers; so convs, linears, activations, etc)
    """
    terminal = {}

    for mod_name, mod in module.named_modules():
        # check if it is a root node (only has itself in named_modules)
        child_count = 0
        for _, __ in mod.named_modules():
            child_count += 1

        if child_count != 1:
            continue

        terminal[mod_name] = mod

    return terminal


def get_conv_layers(module: Module) -> Dict[str, Module]:
    """
    :param module: the module to grab all conv layers for
    :return: a list of all the conv layers in the module
    """
    convs = {}

    for name, mod in module.named_modules():
        if isinstance(mod, _ConvNd):
            convs[name] = mod

    return convs


def get_linear_layers(module: Module) -> Dict[str, Module]:
    """
    :param module: the module to grab all linear layers for
    :return: a list of all linear layers in the module
    """
    linears = {}

    for name, mod in module.named_modules():
        if isinstance(mod, Linear):
            linears[name] = mod

    return linears


def get_prunable_layers(module: Module) -> List[Tuple[str, Module]]:
    """
    :param module: the module to get the prunable layers from
    :return: a list containing the names and modules of the prunable layers
        (Linear, ConvNd)
    """
    layers = []

    for name, mod in module.named_modules():
        if isinstance(mod, Linear) or isinstance(mod, _ConvNd):
            layers.append((name, mod))

    return layers


def get_layer_param(param: str, layer: str, module: Module) -> Parameter:
    """
    :param param: the name of the param to grab from the layer
    :param layer: the name of the layer to grab from the module
    :param module: the module containing the layer and the param
    :return: the param taken from the given layer in the module
    """
    layer = get_layer(layer, module)  # type: Module
    param = layer.__getattr__(param)  # type: Parameter

    return param


def get_named_layers_and_params_by_regex(
    module: Module,
    param_names: List[str],
    params_strict: bool = False,
) -> List[NamedLayerParam]:
    """
    :param module: the module to get the matching layers and params from
    :param param_names: a list of names or regex patterns to match with full parameter
        paths. Regex patterns must be specified with the prefix 're:'
    :param params_strict: if True, this function will raise an exception if there a
        parameter is not found to match every name or regex in param_names
    :return: a list of NamedLayerParam tuples whose full parameter names in the given
        module match one of the given regex patterns or parameter names
    """
    named_layers_and_params = []
    for layer_name, layer in module.named_modules():
        for param_name, param in layer.named_parameters():
            if "." in param_name:  # skip parameters of nested layers
                continue
            full_param_name = "{}.{}".format(layer_name, param_name)
            if any_str_or_regex_matches_param_name(full_param_name, param_names):
                named_layers_and_params.append(
                    NamedLayerParam(layer_name, layer, param_name, param)
                )
    if params_strict:
        validate_all_params_found(param_names, named_layers_and_params)

    return named_layers_and_params


def any_str_or_regex_matches_param_name(
    param_name: str,
    name_or_regex_patterns: List[str],
) -> bool:
    """
    :param param_name: The name of a parameter
    :param name_or_regex_patterns: List of full param names to match to the input or
        regex patterns to match with that should be prefixed with 're:'
    :return: True if any given str or regex pattern matches the given name
    """
    for name_or_regex in name_or_regex_patterns:
        if name_or_regex[:3] == "re:":
            pattern = name_or_regex[3:]
            if re.match(pattern, param_name):
                return True
        else:
            if param_name == name_or_regex:
                return True
    return False


def validate_all_params_found(
    name_or_regex_patterns: List[str],
    named_layers_and_params: List[NamedLayerParam],
):
    """
    :param name_or_regex_patterns: List of full param names or regex patterns of them
        to check for matches in named_layers_and_params names
    :param named_layers_and_params: List of NamedLayerParam objects to check for matches
    :raise RuntimeError: If there is a name or regex pattern that does not have a
        match in named_layers_and_params
    """
    full_parameter_names = [
        "{}.{}".format(named_layer_param.layer_name, named_layer_param.param_name)
        for named_layer_param in named_layers_and_params
    ]
    for name_or_regex in name_or_regex_patterns:
        if "re:" != name_or_regex[:3] and name_or_regex in full_parameter_names:
            continue  # name found in list of full parameter names
        if "re:" == name_or_regex[:3] and any(
            re.match(name_or_regex[3:], name) for name in full_parameter_names
        ):
            continue  # regex pattern matches at least one full parameter name

        raise RuntimeError(
            "All supplied parameter names or regex patterns not found."
            "No match for {} in found parameters {}.  Supplied {}".format(
                name_or_regex, full_parameter_names, name_or_regex_patterns
            )
        )


def set_deterministic_seeds(seed: int = 0):
    """
    Manually seeds the numpy, random, and torch packages.
    Also sets torch.backends.cudnn.deterministic to True
    :param seed: the manual seed to use. Default is 0
    """
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each
    local 0 ranked process to do something.
    :param local_rank: the local rank of this process
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()
