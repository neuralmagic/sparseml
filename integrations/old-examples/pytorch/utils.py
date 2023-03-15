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
from enum import Enum, auto, unique
from typing import Any, List, Optional, Tuple, Union

import torch
from torch.nn import Module
from torch.nn import functional as torch_functional
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from sparseml.pytorch.datasets import DatasetRegistry, ssd_collate_fn, yolo_collate_fn
from sparseml.pytorch.image_classification.utils.helpers import (
    download_framework_model_by_recipe_type,
)
from sparseml.pytorch.models import ModelRegistry
from sparseml.pytorch.optim import ScheduledModifierManager, ScheduledOptimizer
from sparseml.pytorch.sparsification import ConstantPruningModifier
from sparseml.pytorch.utils import (
    DEFAULT_LOSS_KEY,
    CrossEntropyLossWrapper,
    InceptionCrossEntropyLossWrapper,
    LossWrapper,
    ModuleExporter,
    ModuleRunResults,
    PythonLogger,
    SSDLossWrapper,
    TensorBoardLogger,
    TopKAccuracy,
    YoloLossWrapper,
    early_stop_data_loader,
    torch_distributed_zero_first,
)
from sparseml.utils import create_dirs
from sparsezoo import Model


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
    args: Any, task: Optional[Tasks] = None
) -> Tuple[Union[str, None], List]:
    if args.is_main_process:
        save_dir = os.path.abspath(os.path.expanduser(args.save_dir))
        logs_dir = (
            os.path.abspath(os.path.expanduser(os.path.join(args.logs_dir)))
            if task == Tasks.TRAIN
            else None
        )

        if not args.model_tag:
            dataset_name = (
                f"{args.dataset}-{args.dataset_kwargs['year']}"
                if "year" in args.dataset_kwargs
                else args.dataset
            )
            model_tag = f"{args.arch_key.replace('/', '.')}_{dataset_name}"
            model_id = model_tag
            model_inc = 0
            # set location to check for models with same name
            model_main_dir = logs_dir or save_dir

            while os.path.exists(os.path.join(model_main_dir, model_id)):
                model_inc += 1
                model_id = f"{model_tag}__{model_inc:02d}"
        else:
            model_id = args.model_tag

        save_dir = os.path.join(save_dir, model_id)
        create_dirs(save_dir)

        # loggers setup
        loggers = [PythonLogger()]
        if task == Tasks.TRAIN:
            logs_dir = os.path.join(logs_dir, model_id)
            create_dirs(logs_dir)
            loggers.append(TensorBoardLogger(log_path=logs_dir))
        print(f"Model id is set to {model_id}")
    else:
        # do not log for non main processes
        save_dir = None
        loggers = []
    return save_dir, loggers


# data helpers_
def _get_collate_fn(arch_key: str, task: Optional[Tasks] = None):
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

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=args.loader_num_workers,
            pin_memory=args.loader_pin_memory,
            sampler=sampler,
            collate_fn=_get_collate_fn(arch_key=args.arch_key, task=task),
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


def get_train_and_validation_loaders(
    args: Any, image_size: Tuple[int, ...], task: Optional[Tasks] = None
):
    """
    :param args: Object containing relevant configuration for the task
    :param image_size: A Tuple of integers representing the shape of input image
    :param task: The current task being performed
    :return: 4 element tuple with the following format (train_dataset,
        train_loader, val_dataset, val_loader)
    """
    train_dataset, train_loader = _create_train_dataset_and_loader(
        args, image_size, task=task
    )
    val_dataset, val_loader = _create_val_dataset_and_loader(
        args, image_size, task=task
    )
    return train_dataset, train_loader, val_dataset, val_loader


# Model creation Helpers


def create_model(args: Any, num_classes: int) -> Module:
    """
    :param args: object with configuration for model classes
    :param num_classes: Integer representing the number of output classes
    :returns: A Module object representing the created model
    """
    with torch_distributed_zero_first(args.local_rank):  # only download once locally
        if args.checkpoint_path == "zoo":
            if args.recipe_path and args.recipe_path.startswith("zoo:"):
                zoo_model = Model(args.recipe_path)
                args.checkpoint_path = download_framework_model_by_recipe_type(
                    zoo_model
                )
            else:
                raise ValueError(
                    "'zoo' provided as --checkpoint-path but a SparseZoo stub"
                    " prefixed by 'zoo:' not provided as --recipe-path"
                )

        model = ModelRegistry.create(
            args.arch_key,
            args.pretrained,
            args.checkpoint_path,
            args.pretrained_dataset,
            num_classes=num_classes,
            **args.model_kwargs,
        )
    print(f"created model: {model}")
    return model


def infer_num_classes(args: Any, train_dataset, val_dataset):
    """
    :param args: Object with configuration settings
    :param train_dataset: dataset representing training data
    :param val_dataset: dataset representing validation data
    :return: An integer representing the number of classes
    """
    if "num_classes" in args.model_kwargs:
        # handle manually overriden num classes
        num_classes = args.model_kwargs["num_classes"]
        del args.model_kwargs["num_classes"]
    elif args.dataset == "imagefolder":
        dataset = val_dataset or train_dataset  # get non None dataset
        num_classes = dataset.num_classes
    else:
        dataset_attributes = DatasetRegistry.attributes(args.dataset)
        num_classes = dataset_attributes["num_classes"]
    return num_classes


def get_loss_wrapper(
    arch_key: str, training: bool = False, task: Optional[Tasks] = None
):
    """
    :param arch_key: The model architecture
    :param training: True if training task started else False
    :param task: current task being executed
    """
    if "ssd" in arch_key.lower():
        return SSDLossWrapper()
    if "yolo" in arch_key.lower():
        return YoloLossWrapper()

    extras = {"top1acc": TopKAccuracy(1), "top5acc": TopKAccuracy(5)}
    if task == Tasks.TRAIN:
        return (
            CrossEntropyLossWrapper(extras)
            if training and "inception" not in arch_key
            else InceptionCrossEntropyLossWrapper(extras)
        )
    return LossWrapper(loss_fn=torch_functional.cross_entropy, extras=extras)


#  optimizer helper
def create_scheduled_optimizer(
    train_args: Any,
    model: Module,
    train_loader: DataLoader,
    loggers: List[Any],
) -> Tuple[int, ScheduledOptimizer, ScheduledModifierManager]:
    """
    :param train_args : An object with task specific config
    :param model: model architecture to train
    :param train_loader: A DataLoader for training data
    :param loggers: List of loggers to use during training process
    :type train_args: TrainingArguments
    """
    # # optimizer setup
    optim_const = torch.optim.__dict__[train_args.optim]
    optim = optim_const(
        model.parameters(), lr=train_args.init_lr, **train_args.optim_args
    )
    print(f"created optimizer: {optim}")
    print(
        "note, the lr for the optimizer may not reflect the manager yet until "
        "the recipe config is created and run"
    )

    # restore from previous check point
    if train_args.checkpoint_path:
        # currently optimizer restoring is unsupported
        # mapping of the restored params to the correct device is not working
        # load_optimizer(args.checkpoint_path, optim)
        epoch = 0  # load_epoch(args.checkpoint_path) + 1
        print(
            f"restored checkpoint from {train_args.checkpoint_path} for "
            f"epoch {epoch - 1}"
        )
    else:
        epoch = 0

    # modifier setup
    add_mods = (
        ConstantPruningModifier.from_sparse_model(model)
        if train_args.sparse_transfer_learn
        else None
    )
    manager = ScheduledModifierManager.from_yaml(
        file_path=train_args.recipe_path, add_modifiers=add_mods
    )
    optim = ScheduledOptimizer(
        optim,
        model,
        manager,
        steps_per_epoch=len(train_loader),
        loggers=loggers,
    )
    print(f"created manager: {manager}")
    return epoch, optim, manager


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
    input_shape: Tuple[int, ...],
    save_name: str,
    save_dir: str,
    epoch: int,
    val_res: Union[ModuleRunResults, None],
    convert_qat: bool = False,
):
    """
    :param model: model architecture
    :param optim: The optimizer used
    :param input_shape: A tuple of integers representing the input shape
    :param save_name: name to save model to
    :param save_dir: directory to save results in
    :param epoch: integer representing umber of epochs to
    :param val_res: results from validation run
    :param convert_qat: True if model is to be quantized before saving
    """
    has_top1 = "top1acc" in val_res.results
    metric_name = "top-1 accuracy" if has_top1 else "val_loss"
    metric = val_res.result_mean("top1acc" if has_top1 else DEFAULT_LOSS_KEY).item()
    print(
        f"Saving model for epoch {epoch} and {metric_name} "
        f"{metric} to {save_dir} for {save_name}"
    )
    exporter = ModuleExporter(model, save_dir)
    exporter.export_pytorch(optim, epoch, f"{save_name}.pth")
    exporter.export_onnx(
        torch.randn(1, *input_shape),
        f"{save_name}.onnx",
        convert_qat=convert_qat,
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
