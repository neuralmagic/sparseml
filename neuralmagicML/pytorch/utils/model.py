"""
Code related to interacting with a trained model such as saving, loading, etc
"""

from typing import Union, List, Tuple
import torch
from torch.nn import DataParallel, Module
from torch.optim.optimizer import Optimizer
from collections import OrderedDict


__all__ = [
    "load_model",
    "load_optimizer",
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
):
    model_dict = torch.load(path, map_location="cpu")
    current_dict = model.state_dict()

    if "state_dict" in model_dict:
        model_dict = model_dict["state_dict"]

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


def load_optimizer(path: str, optimizer: Optimizer) -> Union[int, None]:
    model_dict = torch.load(path, map_location="cpu")
    optimizer.load_state_dict(model_dict["optimizer"])

    if "epoch" in model_dict:
        return model_dict["epoch"]

    return None


def save_model(
    path: str,
    model: Module,
    optimizer: Optimizer = None,
    epoch: Union[int, None] = None,
):
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
    return _DataParallel(model, ids)


def model_to_device(
    model: Module, device: str
) -> Tuple[Module, str, Union[None, List[int]]]:
    device, ids = device_to_name_ids(device)

    if ids is not None:
        model = parallelize_model(model, ids)

    model = model.to(device)

    return model, device, ids


def device_to_name_ids(device: str) -> Tuple[str, Union[None, List[int]]]:
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
