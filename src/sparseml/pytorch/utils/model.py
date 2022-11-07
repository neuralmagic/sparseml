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
Code related to interacting with a trained model such as saving, loading, etc
"""

from collections import OrderedDict
from typing import Any, List, Optional, Tuple, Union

import torch
from packaging import version
from torch.nn import DataParallel, Module
from torch.optim.optimizer import Optimizer

from sparseml.pytorch.utils.helpers import (
    download_framework_model_by_recipe_type,
    thin_model_from_checkpoint,
)
from sparseml.utils.helpers import create_parent_dirs
from sparsezoo import Model


try:
    from torch.nn.parallel import DistributedDataParallel as DDP

    ddp_import_error = None
except Exception as ddp_error:
    DDP = None
    ddp_import_error = ddp_error


__all__ = [
    "load_model",
    "load_optimizer",
    "load_epoch",
    "save_model",
    "script_model",
    "trace_model",
    "model_to_device",
    "parallelize_model",
    "device_to_name_ids",
    "is_parallel_model",
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

    :param path: the path to the pth file to load the state dict from.
        May also be a SparseZoo stub path preceded by 'zoo:' with the optional
        `?recipe_type=` argument. If given a recipe type, the base model weights
        for that recipe will be loaded.
    :param model: the model to load the state dict into
    :param strict: True to enforce that all tensors match between the model
        and the file; False otherwise
    :param ignore_error_tensors: names of tensors to ignore if they are not found
        in either the model or the file
    :param fix_data_parallel: fix the keys in the model state dict if they
        look like they came from DataParallel type setup (start with module.).
        This removes "module." all keys
    """
    if path.startswith("zoo:"):
        path = download_framework_model_by_recipe_type(Model(path))
    model_dict = torch.load(path, map_location="cpu")
    recipe = model_dict.get("recipe")

    if recipe:
        from sparseml.pytorch.optim import ScheduledModifierManager

        epoch = model_dict.get("epoch", float("inf"))
        checkpoint_manager = ScheduledModifierManager.from_yaml(recipe)
        checkpoint_manager.apply_structure(module=model, epoch=epoch)

    current_dict = model.state_dict()
    if "state_dict" in model_dict:
        model_dict = model_dict["state_dict"]

    # check if any keys were saved through DataParallel type setup and convert those
    if fix_data_parallel:
        keys = [k for k in model_dict.keys()]
        module_key = "module."

        for key in keys:
            if key.startswith(module_key):
                new_key = key[len(module_key) :]
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

    # safety pass for updating layer param shapes when loading a thinned model
    thin_model_from_checkpoint(model, model_dict)

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


def trace_model(
    path: str,
    model: Module,
    sample_batch: Any,
):
    """
    Convenience function which traces the provided module using the sample batch
    into a TorchScript script and saves to provied path.

    :param path: path to save torchscript
    :param model: module to convert to TorchScript
    :param sample_batch: sample batch to trace module with
    """
    script = torch.jit.trace_module(model, {"forward": sample_batch})
    torch.jit.save(script, path)


def script_model(
    path: str,
    model: Module,
):
    """
    Convenience function which scripts the provided module into a TorchScript script
    and saves to provied path.

    :param path: path to save torchscript
    :param model: module to convert to torchscript
    """
    script = torch.jit.script(model)
    torch.jit.save(script, path)


def save_model(
    path: str,
    model: Module,
    optimizer: Optimizer = None,
    recipe: Optional[str] = None,
    epoch: Optional[int] = None,
    use_zipfile_serialization_if_available: bool = True,
    include_modifiers: bool = False,
    arch_key: Optional[str] = None,
):
    """
    Save a model's state dict into a file at the given path.
    Additionally can save an optimizer's state and the current epoch.

    :param path: the path to save the file the states to
    :param model: the model to save state for
    :param optimizer: the optimizer, if any, to save state for
    :param recipe: the recipe used to obtain the model
    :param epoch: the epoch to save
    :param use_zipfile_serialization_if_available: for torch >= 1.6.0 only
        exports the model's state dict using the new zipfile serialization
    :param include_modifiers: if True, and a ScheduledOptimizer is provided
        as the optimizer, the associated ScheduledModifierManager and its
        Modifiers will be exported under the 'manager' key. Default is False
    :param arch_key: if provided, the `arch_key` will be saved in the
        checkpoint
    """
    create_parent_dirs(path)

    if is_parallel_model(model):
        model = model.module

    save_dict = {"state_dict": OrderedDict()}

    # make sure we have the model state_dict on cpu
    for key, state in model.state_dict().items():
        copy = torch.zeros(state.shape)
        copy.copy_(state)
        save_dict["state_dict"][key] = copy

    if optimizer:
        save_dict["optimizer"] = optimizer.state_dict()

    if recipe:
        save_dict["recipe"] = recipe

    if epoch is not None:
        save_dict["epoch"] = epoch

    if include_modifiers and optimizer and hasattr(optimizer, "manager_state_dict"):
        save_dict["manager"] = optimizer.manager_state_dict()
    elif include_modifiers and optimizer and hasattr(optimizer, "wrapped_manager"):
        save_dict["manager"] = optimizer.wrapped_manager.state_dict()

    if arch_key:
        save_dict["arch_key"] = arch_key

    if version.parse(torch.__version__) < version.parse("1.6"):
        torch.save(save_dict, path)
    else:
        torch.save(
            save_dict,
            path,
            _use_new_zipfile_serialization=use_zipfile_serialization_if_available,
        )


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
    model: Module,
    device: Union[str, int],
    ddp: bool = False,
) -> Tuple[Module, str, Union[None, List[int]]]:
    """
    The model to push onto a device or multiple devices.

    :param model: the model to push to a device
    :param device: the device string to push to; ex: cpu, cuda, cuda:0,1. For
        DDP, device should be the local_rank int value; ex: 0
    :param ddp: set True to wrap module as a DDP object. If True, device should
        be set to the local_rank int value. Default is False
    :return: a tuple containing the model on desired device(s),
        the device name, and the ids for the device
    """
    if not ddp:
        device, ids = device_to_name_ids(device)

        if ids is not None:
            model = parallelize_model(model, ids)

        model = model.to(device)
    else:
        if DDP is None:
            raise ddp_import_error
        assert isinstance(
            device, int
        ), "For DDP, device must be set to a local_rank int value"
        assert device < torch.cuda.device_count(), (
            "Device local rank must be less than the number of available cuda devices. "
            "Received local rank {} with device count=={}"
        ).format(device, torch.cuda.device_count())

        model = model.to(device)
        model = DDP(model, device_ids=[device], output_device=device)
        ids = [device]
        device = "cuda:{}".format(device)

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


def is_parallel_model(model: Module) -> bool:
    """
    :param model: the model to test
    :return: True if the given model is wrapped as a DataPararallel or
        DistributedDataParallel Module. False otherwise
    """
    return isinstance(model, DataParallel) or (
        DDP is not None and isinstance(model, DDP)
    )
