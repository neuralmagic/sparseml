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
Helper methods for image classification/detection based tasks
"""
import os
import warnings
from enum import Enum, auto, unique
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from sparseml.pytorch.datasets import DatasetRegistry, ssd_collate_fn, yolo_collate_fn
from sparseml.pytorch.models import ModelRegistry
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import (
    DEFAULT_LOSS_KEY,
    ModuleExporter,
    ModuleRunResults,
    PythonLogger,
    TensorBoardLogger,
    early_stop_data_loader,
    torch_distributed_zero_first,
)
from sparseml.utils import create_dirs
from sparsezoo import Zoo


__all__ = [
    "Tasks",
    "get_save_dir_and_loggers",
    "get_train_and_validation_loaders",
    "create_model",
    "infer_num_classes",
    "save_recipe",
    "save_model_training",
]


@unique
class Tasks(Enum):
    """
    A class representing supported image classification/detection tasks
    """

    TRAIN = auto()
    EXPORT = auto()
    ANALYSIS = auto()
    LR_ANALYSIS = auto()
    PR_SENSITIVITY = auto()


# loggers
def get_save_dir_and_loggers(
    task: Optional[Tasks] = None,
    is_main_process: bool = True,
    save_dir: Optional[str] = None,
    logs_dir: Optional[str] = None,
    arch_key: Optional[str] = "",
    model_tag: Optional[str] = None,
    dataset_name: Optional[str] = "",
) -> Tuple[Union[str, None], List]:
    """
    :param task: The current task being performed
    :param is_main_process: Whether this is the main process or not
    :param save_dir: The directory to save the model
    :param logs_dir: The directory to save logs
    :param arch_key: The architecture key of the image classification model
    :param model_tag: A str tag to optionally tag this model with
    :param dataset_name: The name of the dataset used to tag a model
        if model_tag not given
    :return: A tuple of the save directory and a list of loggers
    """
    if is_main_process:
        save_dir = os.path.abspath(os.path.expanduser(save_dir))
        logs_dir = (
            os.path.abspath(os.path.expanduser(os.path.join(logs_dir)))
            if task == Tasks.TRAIN
            else None
        )

        if not model_tag:
            model_tag = f"{arch_key.replace('/', '.')}_{dataset_name}"
            model_id = model_tag
            model_inc = 0
            # set location to check for models with same name
            model_main_dir = logs_dir or save_dir

            while os.path.exists(os.path.join(model_main_dir, model_id)):
                model_inc += 1
                model_id = f"{model_tag}__{model_inc:02d}"
        else:
            model_id = model_tag

        save_dir = os.path.join(save_dir, model_id)
        create_dirs(save_dir)

        # loggers setup
        loggers = [PythonLogger()]
        if task == Tasks.TRAIN:
            logs_dir = os.path.join(logs_dir, model_id)
            create_dirs(logs_dir)
            try:
                loggers.append(TensorBoardLogger(log_path=logs_dir))
            except AttributeError:
                warnings.warn(
                    "Failed to initialize TensorBoard logger, "
                    "it will not be used for logging",
                )

        print(f"Model id is set to {model_id}")
    else:
        # do not log for non main processes
        save_dir = None
        loggers = []
    return save_dir, loggers


# data helpers
def get_train_and_validation_loaders(args: Any, task: Optional[Tasks] = None):
    """
    :param args: Object containing relevant configuration for the task
    :param task: The current task being performed
    :return: 4 element tuple with the following format (train_dataset,
        train_loader, val_dataset, val_loader)
    """
    train_dataset, train_loader = _create_train_dataset_and_loader(
        args, args.image_size, task=task
    )
    val_dataset, val_loader = _create_val_dataset_and_loader(
        args, args.image_size, task=task
    )
    return train_dataset, train_loader, val_dataset, val_loader


# Model creation Helpers
def create_model(args: Any, num_classes: int) -> Tuple[Module, str]:
    """
    :param args: object with configuration for model classes
    :param num_classes: Integer representing the number of output classes
    :returns: A tuple containing the model and the model's arch_key
    """
    with torch_distributed_zero_first(args.local_rank):
        # only download once locally
        if args.checkpoint_path == "zoo":
            args.checkpoint_path = _download_model_from_zoo_using_recipe(
                recipe_stub=args.recipe_path,
            )

        result = ModelRegistry.create(
            key=args.arch_key,
            pretrained=args.pretrained,
            pretrained_path=args.checkpoint_path,
            pretrained_dataset=args.pretrained_dataset,
            num_classes=num_classes,
            **args.model_kwargs,
        )

        if not isinstance(result, tuple):
            model, arch_key = result, args.arch_key
        else:
            model, arch_key = result

        return model, arch_key


def infer_num_classes(
    train_dataset, val_dataset, dataset: str, model_kwargs: Dict[str, Any]
) -> int:
    """
    :param train_dataset: dataset representing training data
    :param val_dataset: dataset representing validation data
    :param dataset: name of the dataset
    :param model_kwargs: keyword arguments used for model creation
    :return: An integer representing the number of classes
    """
    if "num_classes" in model_kwargs:
        # handle manually overriden num classes
        num_classes = model_kwargs["num_classes"]
        del model_kwargs["num_classes"]
    elif dataset == "imagefolder":
        dataset = val_dataset or train_dataset  # get non None dataset
        num_classes = dataset.num_classes
    else:
        dataset_attributes = DatasetRegistry.attributes(dataset)
        num_classes = dataset_attributes["num_classes"]
    return num_classes


def get_arch_key(arch_key: Optional[str], checkpoint_path: Optional[str]) -> str:
    """
    Utility method to read and return the arch_key from the checkpoint, if it
    is not passed and exists in the checkpoint. if passed the passed value is
    returned

    :param arch_key: Optional[str] The arch_key to use for the model
    :checkpoint_path: Optiona[str] The path to the checkpoint
    """
    if arch_key is None:
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
        else:
            raise ValueError(
                "Must provide a checkpoint path if no arch_key is provided"
            )
        if "arch_key" in checkpoint:
            arch_key = checkpoint["arch_key"]
        else:
            raise ValueError(
                "Checkpoint does not contain "
                "arch_key, provide one using "
                "--arch_key"
            )

    return arch_key


# saving helpers


def save_recipe(
    recipe_manager: ScheduledModifierManager,
    save_dir: str,
):
    """
    :param recipe_manager: The ScheduleModified manager to save recipes
    :param save_dir: The directory to save the recipe
    """
    recipe_save_path = os.path.join(save_dir, "recipe.yaml")
    recipe_manager.save(recipe_save_path)
    print(f"Saved recipe to {recipe_save_path}")


def save_model_training(
    model: Module,
    optim: Optimizer,
    save_name: str,
    save_dir: str,
    epoch: int,
    val_res: Optional[ModuleRunResults],
    arch_key: Optional[str] = None,
):
    """
    :param model: model architecture
    :param optim: The optimizer used
    :param save_name: name to save model to
    :param save_dir: directory to save results in
    :param epoch: integer representing umber of epochs to
    :param val_res: results from validation run
    :param arch_key: if provided, the `arch_key` will be saved in the
        checkpoint
    """
    has_top1 = "top1acc" in val_res.results
    metric_name = "top-1 accuracy" if has_top1 else "val_loss"
    metric = val_res.result_mean("top1acc" if has_top1 else DEFAULT_LOSS_KEY).item()
    print(
        f"Saving model for epoch {epoch} and {metric_name} "
        f"{metric} to {save_dir} for {save_name}"
    )
    exporter = ModuleExporter(model, save_dir)
    exporter.export_pytorch(
        optim,
        epoch,
        f"{save_name}.pth",
        arch_key=arch_key,
    )
    info_path = os.path.join(save_dir, f"{save_name}.txt")

    with open(info_path, "w") as info_file:
        info_lines = [
            f"epoch: {epoch}",
        ]

        if val_res is not None:
            for loss in val_res.results.keys():
                info_lines.append(f"{loss}: {val_res.result_mean(loss).item()}")

        info_file.write("\n".join(info_lines))


# private methods
def _get_collate_fn(arch_key: Optional[str] = None, task: Optional[Tasks] = None):
    if not arch_key:
        return None

    is_ssd = "ssd" in arch_key.lower()
    need_collate_function = (
        is_ssd or "yolo" in arch_key.lower()
    ) and task != Tasks.EXPORT
    if need_collate_function:
        return ssd_collate_fn if is_ssd else yolo_collate_fn
    return None


def _create_train_dataset_and_loader(
    args: Any,
    image_size: Tuple[int, ...],
    task: Optional[Tasks] = None,
) -> Tuple[Any, Any]:
    need_train_data = not (
        task == Tasks.EXPORT
        or task == Tasks.PR_SENSITIVITY
        and args.approximate
        or task == Tasks.TRAIN
        and args.eval_mode
    )

    if need_train_data:
        with torch_distributed_zero_first(
            args.local_rank,
        ):  # only download once locally
            train_dataset = DatasetRegistry.create(
                args.dataset,
                root=args.dataset_path,
                train=True,
                rand_trans=True,
                image_size=image_size,
                **args.dataset_kwargs,
            )
        sampler = (
            torch.utils.data.distributed.DistributedSampler(train_dataset)
            if args.rank != -1
            else None
        )
        shuffle = True if sampler is None else False
        batch_size = args.train_batch_size if task == Tasks.TRAIN else args.batch_size

        arch_key = args.arch_key if hasattr(args, "arch_key") else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=args.loader_num_workers,
            pin_memory=args.loader_pin_memory,
            sampler=sampler,
            collate_fn=_get_collate_fn(arch_key=arch_key, task=task),
        )
        print(f"created train_dataset: {train_dataset}")
        return train_dataset, train_loader

    return None, None


def _create_val_dataset_and_loader(
    args, image_size: Tuple[int, ...], task: Optional[Tasks] = None
) -> Tuple[Any, Any]:
    need_val_data = not (
        task == Tasks.PR_SENSITIVITY
        or task == Tasks.LR_ANALYSIS
        or not (args.is_main_process and args.dataset != "imagefolder")
    )

    if need_val_data:
        val_dataset = DatasetRegistry.create(
            args.dataset,
            root=args.dataset_path,
            train=False,
            rand_trans=False,
            image_size=image_size,
            **args.dataset_kwargs,
        )
        if args.is_main_process:
            is_training = task == Tasks.TRAIN
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.test_batch_size if is_training else 1,
                shuffle=False,
                num_workers=args.loader_num_workers if is_training else 1,
                pin_memory=args.loader_pin_memory if is_training else False,
                collate_fn=_get_collate_fn(arch_key=args.arch_key, task=task),
            )
            if task == Tasks.EXPORT:
                val_loader = early_stop_data_loader(
                    val_loader, args.num_samples if args.num_samples > 1 else 1
                )
            print(f"created val_dataset: {val_dataset}")
        else:
            val_loader = None  # only val dataset needed to get the number of classes
        return val_dataset, val_loader
    return None, None  # val dataset not needed


def _download_model_from_zoo_using_recipe(
    recipe_stub: str,
) -> Optional[str]:
    """
    Download a model from the zoo using a recipe stub and return the
    path to the downloaded model.

    :param recipe_stub: Path to a valid recipe stub
    :return: Path to the downloaded model
    """
    valid_recipe_stub = recipe_stub and recipe_stub.startswith("zoo:")

    if not valid_recipe_stub:
        raise ValueError(
            "The recipe path must start with 'zoo:' to download from the zoo"
            f" but got {recipe_stub} instead"
        )

    files = Zoo.download_recipe_base_framework_files(recipe_stub, extensions=[".pth"])

    checkpoint_path = files[0]
    return checkpoint_path
