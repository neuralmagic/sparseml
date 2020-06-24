"""
Code related to interacting with a trained model such as saving, loading, etc
"""

from typing import Union, List, Tuple
import torch
from torch.nn import DataParallel, Module
from torch.optim.optimizer import Optimizer
from collections import OrderedDict

from neuralmagicML.utils.helpers import create_parent_dirs


__all__ = [
    "load_model",
    "load_optimizer",
    "load_epoch",
    "save_model",
    "model_to_device",
    "parallelize_model",
    "device_to_name_ids",
]


def load_model(
    path: str,
    model: Module,
    strict: bool = False,
    ignore_error_tensors: List[str] = None,
    fix_data_parallel: bool = True,
):
    """
    Load the state dict into a model from a given file.

    :param path: the path to the pth file to load the state dict from
    :param model: the model to load the state dict into
    :param strict: True to enforce that all tensors match between the model
        and the file; False otherwise
    :param ignore_error_tensors: names of tensors to ignore if they are not found
        in either the model or the file
    :param fix_data_parallel: fix the keys in the model state dict if they
        look like they came from DataParallel type setup (start with module.).
        This removes "module." all keys
    """
    model_dict = torch.load(path, map_location="cpu")
    current_dict = model.state_dict()

    if "state_dict" in model_dict:
        model_dict = model_dict["state_dict"]

    # check if any keys were saved through DataParallel type setup and convert those
    if fix_data_parallel:
        keys = [k for k in model_dict.keys()]
        module_key = "module."

        for key in keys:
            if key.startswith(module_key):
                new_key = key[len(module_key):]
                model_dict[new_key] = model_dict[key]
                del model_dict[key]

    if not ignore_error_tensors:
        ignore_error_tensors = []

    for ignore in ignore_error_tensors:
        if ignore not in model_dict and ignore not in current_dict:
            continue

        if (
            ignore in model_dict
            and ignore in current_dict
            and current_dict[ignore].shape != model_dict[ignore].shape
        ):
            model_dict[ignore] = current_dict[ignore]
        elif ignore not in model_dict and ignore in current_dict:
            model_dict[ignore] = current_dict[ignore]
        elif ignore in model_dict and ignore not in current_dict:
            del model_dict[ignore]

    model.load_state_dict(model_dict, strict)


def load_optimizer(
    path: str, optimizer: Optimizer, map_location: Union[None, str] = "cpu"
):
    """
    Load the state dict into an optimizer from a given file.

    :param path: the path to the pth file to load the state dict from
    :param optimizer: the optimizer to load the state dict into
    :param map_location: the location to map the values to when loading the
    :return: the epoch saved in the file, if any
    """
    model_dict = torch.load(path, map_location=map_location)
    optimizer.load_state_dict(model_dict["optimizer"])


def load_epoch(path: str, map_location: Union[None, str] = "cpu") -> Union[int, None]:
    model_dict = torch.load(path, map_location=map_location)

    if "epoch" in model_dict:
        return model_dict["epoch"]

    return None


def save_model(
    path: str,
    model: Module,
    optimizer: Optimizer = None,
    epoch: Union[int, None] = None,
):
    """
    Save a model's state dict into a file at the given path.
    Additionally can save an optimizer's state and the current epoch.

    :param path: the path to save the file the states to
    :param model: the model to save state for
    :param optimizer: the optimizer, if any, to save state for
    :param epoch: the epoch to save
    """
    create_parent_dirs(path)

    if isinstance(model, DataParallel):
        model = model.module

    save_dict = {"state_dict": OrderedDict()}

    # make sure we have the model state_dict on cpu
    for key, state in model.state_dict().items():
        copy = torch.zeros(state.shape)
        copy.copy_(state)
        save_dict["state_dict"][key] = copy

    if optimizer:
        save_dict["optimizer"] = optimizer.state_dict()

    if epoch:
        save_dict["epoch"] = epoch

    torch.save(save_dict, path)


class _DataParallel(DataParallel):
    def __getattr__(self, item):
        module = super().__getattr__("module")

        if item == "module":
            return module

        return getattr(module, item)


def parallelize_model(model: Module, ids: Union[None, List[int]]) -> Module:
    """
    Data parallelize a model across multiple devices

    :param model: the model to parallelize across multiple devices
    :param ids: the ides of the devices to parallelize across
    :return: a parallelized model
    """
    return _DataParallel(model, ids)


def model_to_device(
    model: Module, device: str
) -> Tuple[Module, str, Union[None, List[int]]]:
    """
    The model to push onto a device or multiple devices.

    :param model: the model to push to a device
    :param device: the device string to push to; ex: cpu, cuda, cuda:0,1
    :return: a tuple containing the model on desired device(s),
        the device name, and the ids for the device
    """

    device, ids = device_to_name_ids(device)

    if ids is not None:
        model = parallelize_model(model, ids)

    model = model.to(device)

    return model, device, ids


def device_to_name_ids(device: str) -> Tuple[str, Union[None, List[int]]]:
    """
    Split a device string into a device and ids

    :param device: the device string to push to; ex: cpu, cuda, cuda:0,1
    :return: a tuple containing the device string and devices
    """
    split = device.split(":")
    name = split[0]

    if name == "cpu":
        return name, None

    if name != "cuda" or not torch.cuda.is_available():
        raise ValueError("{} device not available on this system".format(name))

    if len(split) < 2:
        return name, None

    ids = [int(id_) for id_ in split[1].split(",")]
    count = torch.cuda.device_count()

    for id_ in ids:
        if id_ >= count:
            raise ValueError("{} device id not available on this system".format(id_))

    if len(ids) == 1:
        return "{}:{}".format(name, ids[0]), None

    return name, ids
