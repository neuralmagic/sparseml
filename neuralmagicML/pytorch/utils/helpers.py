"""
Utility / helper functions
"""

from typing import Union, Tuple, Iterable, Dict, Any, List
import os
import random
import numpy

import torch
from torch import Tensor
from torch.nn import Module, Linear, Parameter
from torch.nn.modules.conv import _ConvNd

from neuralmagicML.utils import create_dirs


__all__ = [
    "tensors_batch_size",
    "tensors_to_device",
    "tensors_to_precision",
    "tensors_module_forward",
    "tensor_export",
    "tensors_export",
    "tensor_density",
    "tensor_sparsity",
    "tensor_sample",
    "abs_threshold_from_sparsity",
    "sparsity_mask_from_abs_threshold",
    "sparsity_mask",
    "sparsity_mask_from_tensor",
    "mask_difference",
    "get_layer",
    "get_terminal_layers",
    "get_conv_layers",
    "get_linear_layers",
    "get_prunable_layers",
    "get_layer_param",
]


##############################
#
# pytorch tensor helper functions
#
##############################


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

    if isinstance(tensors, Tuple):
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

    if isinstance(tensors, Tuple):
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
        (isinstance(tensors, Tuple) or isinstance(tensors, List))
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
    tensor: Union[Tensor, Iterable[Tensor]],
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
    create_dirs(export_dir)
    export_path = os.path.join(
        export_dir, "{}.{}".format(name, "npz" if npz else "npy")
    )

    if isinstance(tensor, Tensor):
        tensor = tensor.detach().cpu().numpy()
        if npz:
            numpy.savez_compressed(export_path, tensor)
        else:
            numpy.save(export_path, tensor)
    else:
        tensor = [tens.detach().cpu().numpy() for tens in tensor]
        if npz:
            numpy.savez_compressed(export_path, *tensor)
        else:
            numpy.save(export_path, tensor)

    return export_path


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
                tens, export_dir, name_prefix, counter + index, exported_paths,
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


def abs_threshold_from_sparsity(tens: Tensor, sparsity: float) -> Tensor:
    """
    :param tens: the tensor to find a value in for which setting
        abs(all values) < that value will give desired sparsity
    :param sparsity: the desired sparsity to apply
    :return: the threshold to get to the desired sparsity or an empty tensor
        if it was not possible given the inputs
    """
    if tens.numel() < 1 or sparsity <= 0.0 or sparsity > 1.0:
        return tens.new_tensor([])

    sorted_vals, _ = torch.sort(tens.abs().view(-1))
    lookup_index = round(sparsity * (tens.numel() - 1))

    if lookup_index < 0:
        lookup_index = 0
    elif lookup_index > tens.numel():
        lookup_index = tens.numel()

    return sorted_vals[lookup_index]


def sparsity_mask(tens: Tensor, sparsity: float) -> Tensor:
    """
    :param tens: the tensor to calculate a mask from based on the contained values
    :param sparsity: the desired sparsity to reach within the mask
        (decimal fraction of zeros)
    :return: a mask (0.0 for values that are masked, 1.0 for values that are unmasked)
        calculated from the tens such that the desired number of zeros
        matches the sparsity. removes the abs lowest values if there are more zeros
        in the tens than desired sparsity, then will randomly choose the zeros
    """
    threshold = abs_threshold_from_sparsity(tens, sparsity)

    if threshold.numel() < 1:
        return tens.new_ones(tens.shape)

    if threshold.item() > 0.0:
        return sparsity_mask_from_abs_threshold(tens, threshold)

    # too many zeros so will go over the already given sparsity
    # and choose which zeros to not keep in mask at random
    zero_indices = (tens == 0.0).nonzero()
    rand_indices = list(range(zero_indices.shape[0]))
    random.shuffle(rand_indices)
    num_elem = tens.numel()
    num_mask = int(num_elem * sparsity)
    rand_indices = rand_indices[:num_mask]
    rand_indices = tens.new_tensor(rand_indices, dtype=torch.int64)
    zero_indices = zero_indices[rand_indices, :]
    mask = tens.new_ones(tens.shape).type(tens.type())
    mask[zero_indices.split(1, dim=1)] = 0

    return mask.type(tens.type())


def sparsity_mask_from_tensor(tens: Tensor) -> Tensor:
    """
    :param tens: the tensor to calculate a mask from the contained values
    :return: a mask (0.0 for values that are masked, 1.0 for values that are unmasked)
        calculated from the tens current values of 0.0 are masked
        and everything else is unmasked
    """
    return torch.ne(tens, 0.0).type(tens.type())


def sparsity_mask_from_abs_threshold(
    tens: Tensor, threshold: Union[float, Tensor]
) -> Tensor:
    """
    :param tens: the tensor to calculate a mask from based on the contained values
    :param threshold: a threshold at which to mask abs(values) if they are
        less than it or equal
    :return: a mask (0.0 for values that are masked, 1.0 for values that are unmasked)
        calculated from the tens abs(values) <= threshold are masked,
        all others are unmasked
    """
    return (torch.abs(tens) > threshold).type(tens.type())


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
    :param module:
    :return:
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
