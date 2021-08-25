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
import argparse
import json
import os
from contextlib import suppress
from enum import Enum, auto, unique
from typing import Any, List, Optional, Tuple, Union

import torch
from torch.nn import Module
from torch.nn import functional as torch_functional
from torch.optim import SGD, Adam, Optimizer
from torch.utils.data import DataLoader

from sparseml import get_main_logger
from sparseml.pytorch.datasets import DatasetRegistry, ssd_collate_fn, yolo_collate_fn
from sparseml.pytorch.models import ModelRegistry
from sparseml.pytorch.optim import (
    ConstantPruningModifier,
    ScheduledModifierManager,
    ScheduledOptimizer,
)
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
    default_device,
    early_stop_data_loader,
    set_deterministic_seeds,
    torch_distributed_zero_first,
)
from sparseml.utils import create_dirs
from sparsezoo import Zoo


with suppress(Exception):
    from torch.optim import RMSprop

LOGGER = get_main_logger()


@unique
class Tasks(Enum):
    """
    A class representing supported image classification/detection tasks
    """

    TRAIN = auto()
    EXPORT = auto()
    ANALYSIS = auto()
    LR_SENSITIVITY = "lr_sensitivity"
    PR_SENSITIVITY = "pr_sensitivity"


def _add_pin_memory_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--loader-pin-memory",
        type=bool,
        default=True,
        help="Use pinned memory for data loading",
    )


def _add_workers_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--loader-num-workers",
        type=int,
        default=4,
        help="The number of workers to use for data loading",
    )


def _add_device_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--device",
        type=str,
        default=default_device(),
        help=(
            "The device to run on (can also include ids for data parallel), ex:"
            " cpu, cuda, cuda:0,1"
        ),
    )


def add_device_memory_and_worker_args(parser: argparse.ArgumentParser):
    _add_device_args(parser)
    _add_workers_args(parser)
    _add_pin_memory_args(parser)


def add_universal_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--arch-key",
        type=str,
        required=True,
        help="The type of model to use, ex: resnet50, vgg16, mobilenet "
        "put as help to see the full list (will raise an exception with the list)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=True,
        help="The type of pretrained weights to use, "
        "default is true to load the default pretrained weights for the model. "
        "Otherwise should be set to the desired weights type: "
        "[base, optim, optim-perf]. "
        "To not load any weights set to one of [none, false]",
    )
    parser.add_argument(
        "--pretrained-dataset",
        type=str,
        default=None,
        help="The dataset to load pretrained weights for if pretrained is set. "
        "Default is None which will load the default dataset for the architecture."
        " Ex can be set to imagenet, cifar10, etc",
    )

    parser.add_argument(
        "--model-kwargs",
        type=json.loads,
        default={},
        help="Keyword arguments to be passed to model constructor, should be "
        " given as a json object",
    )
    # dataset args
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The dataset to use for training, "
        "ex: imagenet, imagenette, cifar10, etc. "
        "Set to imagefolder for a generic dataset setup "
        "with an image folder structure setup like imagenet or loadable by a "
        "dataset in sparseml.pytorch.datasets",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="The root path to where the dataset is stored",
    )
    parser.add_argument(
        "--dataset-kwargs",
        type=json.loads,
        default={},
        help="Keyword arguments to be passed to dataset constructor, should be "
        " given as a json object",
    )
    # logging and saving
    parser.add_argument(
        "--model-tag",
        type=str,
        default=None,
        help="A tag to use for the model for saving results under save-dir, "
        "defaults to the model arch and dataset used",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="pytorch_vision",
        help="The path to the directory for saving results",
    )


def append_preprocessing_args(args: argparse.Namespace):
    if "preprocessing_type" not in args.dataset_kwargs and (
        "coco" in args.dataset.lower() or "voc" in args.dataset.lower()
    ):
        if "ssd" in args.arch_key.lower():
            args.dataset_kwargs["preprocessing_type"] = "ssd"
        elif "yolo" in args.arch_key.lower():
            args.dataset_kwargs["preprocessing_type"] = "yolo"


def distributed_setup(local_rank: int):
    """
    :param local_rank: -1 for distributed training
    """
    # initialize DDP process, set deterministic seeds
    if local_rank != -1:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        set_deterministic_seeds(0)


# loggers


def get_save_dir_and_loggers(
    args: argparse.Namespace, task: Optional[Tasks] = None
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
                args.dataset
                if "year" not in args.dataset_kwargs
                else "{}-{}".format(args.dataset, args.dataset_kwargs["year"])
            )
            model_tag = "{}_{}".format(args.arch_key.replace("/", "."), dataset_name)
            model_id = model_tag
            model_inc = 0
            # set location to check for models with same name
            model_main_dir = logs_dir or save_dir

            while os.path.exists(os.path.join(model_main_dir, model_id)):
                model_inc += 1
                model_id = "{}__{:02d}".format(model_tag, model_inc)
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
        LOGGER.info("Model id is set to {}".format(model_id))
    else:
        # do not log for non main processes
        save_dir = None
        loggers = []
    return save_dir, loggers


# data helpers


def _get_collate_fn(args: argparse.Namespace, task: Optional[Tasks] = None):
    if (
        "ssd" not in args.arch_key.lower() and "yolo" not in args.arch_key.lower()
    ) or task == Tasks.EXPORT:
        # default collate function
        return None
    return ssd_collate_fn if "ssd" in args.arch_key.lower() else yolo_collate_fn


def _create_train_dataset_and_loader(
    args: argparse.Namespace,
    image_size: Tuple[int, ...],
    task: Optional[Tasks] = None,
) -> Tuple[Any, Any]:
    # create train dataset if it should be ran for this flow, otherwise return None
    if (
        task == Tasks.EXPORT
        or task == Tasks.PR_SENSITIVITY
        and args.approximate
        or task == Tasks.TRAIN
        and args.eval_mode
    ):
        return None, None

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
        collate_fn=_get_collate_fn(args),
    )
    LOGGER.info("created train_dataset: {}".format(train_dataset))
    return train_dataset, train_loader


def _create_val_dataset_and_loader(
    args, image_size: Tuple[int, ...], task: Optional[Tasks] = None
) -> Tuple[Any, Any]:
    if (
        task == Tasks.PR_SENSITIVITY
        or task == Tasks.LR_SENSITIVITY
        or not (args.is_main_process and args.dataset != "imagefolder")
    ):
        return None, None  # val dataset not needed
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
            collate_fn=_get_collate_fn(args),
        )
        if task == Tasks.EXPORT:
            val_loader = early_stop_data_loader(
                val_loader, args.num_samples if args.num_samples > 1 else 1
            )
        LOGGER.info("created val_dataset: {}".format(val_dataset))
    else:
        val_loader = None  # only val dataset needed to get the number of classes
    return val_dataset, val_loader


def get_train_and_validation_loaders(
    args, image_size: Tuple[int, ...], task: Optional[Tasks] = None
):
    """
    :param args: Namespace object containing relevant configuration for the task
    :param image_size: A Tuple of integers representing the shape of input image
    :param task: The current task being performed
    :return: 4 element tuple with the following format (train_dataset, train_loader,
        val_dataset, val_loader)
    """
    train_dataset, train_loader = _create_train_dataset_and_loader(
        args, image_size, task=task
    )
    val_dataset, val_loader = _create_val_dataset_and_loader(
        args, image_size, task=task
    )
    return train_dataset, train_loader, val_dataset, val_loader


# Model creation Helpers


def create_model(args: argparse.Namespace, num_classes: int) -> Module:
    """
    :param args: Namespace object with configuration for model classes
    :param num_classes: Integer representing the number of output classes
    """
    with torch_distributed_zero_first(args.local_rank):  # only download once locally
        if args.checkpoint_path == "zoo":
            if args.recipe_path and args.recipe_path.startswith("zoo:"):
                args.checkpoint_path = Zoo.download_recipe_base_framework_files(
                    args.recipe_path, extensions=[".pth"]
                )[0]
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
    LOGGER.info("created model: {}".format(model))
    return model


def infer_num_classes(args, train_dataset, val_dataset):
    """
    :param args: Namespace object with configuration settings
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


def get_loss_wrapper(args, training=False, task: Optional[Tasks] = None):
    """
    :param args: Namespace object with config
    :param training: True if training task started else False
    :param task: current task being executed
    """
    if "ssd" in args.arch_key.lower():
        return SSDLossWrapper()
    if "yolo" in args.arch_key.lower():
        return YoloLossWrapper()

    extras = {"top1acc": TopKAccuracy(1), "top5acc": TopKAccuracy(5)}
    if task == Tasks.TRAIN:
        return (
            CrossEntropyLossWrapper(extras)
            if training and "inception" not in args.arch_key
            else InceptionCrossEntropyLossWrapper(extras)
        )
    return LossWrapper(loss_fn=torch_functional.cross_entropy, extras=extras)


#  optimizer helper
def create_scheduled_optimizer(
    args,
    model: Module,
    train_loader: DataLoader,
    loggers: List[Any],
) -> Tuple[int, ScheduledOptimizer, ScheduledModifierManager]:
    """
    :params args : A Namespace object with task specific config
    :param model: model architecture to train
    :param train_loader: A DataLoader for training data
    :param loggers: List of loggers to use during training process
    """
    # optimizer setup
    if args.optim == "SGD":
        optim_const = SGD
    elif args.optim == "Adam":
        optim_const = Adam
    elif args.optim == "RMSProp":
        optim_const = RMSprop
    else:
        raise ValueError(
            "unsupported value given for optim_type of {}".format(args.optim_type)
        )

    optim = optim_const(model.parameters(), lr=args.init_lr, **args.optim_args)
    LOGGER.info("created optimizer: {}".format(optim))
    LOGGER.info(
        "note, the lr for the optimizer may not reflect the manager yet until "
        "the recipe config is created and run"
    )

    # restore from previous check point
    if args.checkpoint_path:
        # currently optimizer restoring is unsupported
        # mapping of the restored params to the correct device is not working
        # load_optimizer(args.checkpoint_path, optim)
        epoch = 0  # load_epoch(args.checkpoint_path) + 1
        LOGGER.info(
            "restored checkpoint from {} for epoch {}".format(
                args.checkpoint_path, epoch - 1
            )
        )
    else:
        epoch = 0

    # modifier setup
    add_mods = (
        ConstantPruningModifier.from_sparse_model(model)
        if args.sparse_transfer_learn
        else None
    )
    manager = ScheduledModifierManager.from_yaml(
        file_path=args.recipe_path, add_modifiers=add_mods
    )
    optim = ScheduledOptimizer(
        optim,
        model,
        manager,
        steps_per_epoch=len(train_loader),
        loggers=loggers,
    )
    LOGGER.info("created manager: {}".format(manager))
    return epoch, optim, manager


# saving helpers


def save_recipe(
    recipe_manager: ScheduledModifierManager,
    save_dir: str,
):
    recipe_save_path = os.path.join(save_dir, "recipe.yaml")
    recipe_manager.save(recipe_save_path)
    LOGGER.info(f"Saved recipe to {recipe_save_path}")


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
    :param model: model architecture to train
    :param optim: The optimizer used
    :param input_shape: A tuple of integers representing the input shape
    :param save_name: name to save model to
    :param save_dir: directory to save results in
    :param epoch: integer representing umber of epochs to
    :param val_res: results from validation run
    :param convert_qat: True if model is to be quantized before saving
    """
    LOGGER.info(
        "Saving model for epoch {} and val_loss {} to {} for {}".format(
            epoch, val_res.result_mean(DEFAULT_LOSS_KEY).item(), save_dir, save_name
        )
    )
    exporter = ModuleExporter(model, save_dir)
    exporter.export_pytorch(optim, epoch, "{}.pth".format(save_name))
    exporter.export_onnx(
        torch.randn(1, *input_shape),
        "{}.onnx".format(save_name),
        convert_qat=convert_qat,
    )

    info_path = os.path.join(save_dir, "{}.txt".format(save_name))

    with open(info_path, "w") as info_file:
        info_lines = [
            "epoch: {}".format(epoch),
        ]

        if val_res is not None:
            for loss in val_res.results.keys():
                info_lines.append(
                    "{}: {}".format(loss, val_res.result_mean(loss).item())
                )

        info_file.write("\n".join(info_lines))


def _set_ddp_deafaults(args):
    # set ddp args to default values
    args.local_rank = -1
    args.rank = -1
    args.world_size = 1
    args.is_main_process = True


def append_ddp_defaults_and_preprocessing_args(args):
    _set_ddp_deafaults(args=args)
    append_preprocessing_args(args=args)
