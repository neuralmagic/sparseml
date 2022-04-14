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
from contextlib import contextmanager
from enum import Enum, auto, unique
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from sparseml.pytorch.datasets import DatasetRegistry
from sparseml.pytorch.datasets.image_classification.ffcv_dataset import (
    FFCVCompatibleDataset,
)
from sparseml.pytorch.models import ModelRegistry
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import (
    DEFAULT_LOSS_KEY,
    ModuleExporter,
    ModuleRunResults,
    PythonLogger,
    TensorBoardLogger,
    default_device,
    early_stop_data_loader,
    torch_distributed_zero_first,
)
from sparseml.utils import create_dirs
from sparsezoo import Zoo


__all__ = [
    "Tasks",
    "get_save_dir_and_loggers",
    "get_dataset_and_dataloader",
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
def get_dataset_and_dataloader(
    dataset_name: str,
    dataset_path: str,
    batch_size: int,
    image_size: int,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
    training: bool = False,
    rank: int = -1,
    local_rank: int = -1,
    loader_num_workers: int = 0,
    loader_pin_memory: bool = False,
    max_samples: Optional[int] = None,
    ffcv: bool = False,
    device: Optional[torch.device] = default_device(),
) -> Tuple[Dataset, Union[DataLoader, Any]]:
    """
    :param dataset_name: The name of the dataset
    :param dataset_path: The path to the dataset
    :param batch_size: The batch size
    :param image_size: A tuple of ints representing the image size
    :param dataset_kwargs: A dict of kwargs for dataset creation
    :param training: Whether this is training or validation
    :param rank: The rank of the current process
    :param local_rank: The local rank of the current process
    :param loader_num_workers: The number of workers to use for the data loader
    :param loader_pin_memory: Whether to pin memory for the data loader
    :param max_samples: The maximum number of samples to use
    :param ffcv: Whether to use ffcv dataset and data loaders
    :param device: The device to use for the data loader. Required for ffcv
    :return: Tuple with the following format (dataset, dataloader)
    """

    download_context = (
        torch_distributed_zero_first(local_rank)  # only download once locally
        if training
        else _nullcontext()
    )
    dataset_kwargs = dataset_kwargs or {}

    with download_context:
        dataset = DatasetRegistry.create(
            dataset_name,
            root=dataset_path,
            train=training,
            rand_trans=training,
            image_size=image_size,
            **dataset_kwargs,
        )

    sampler = (
        torch.utils.data.distributed.DistributedSampler(dataset)
        if rank != -1 and training  # only run on DDP + training
        else None
    )
    shuffle = sampler is None and not training

    if ffcv:
        if not isinstance(dataset, FFCVCompatibleDataset):
            raise ValueError(f"Dataset {dataset} must implement FFCVCompatibleDataset")
        dataset_type = "train" if training else "val"
        write_path = os.path.join(
            dataset_path,
            "ffcv_cache",
            f"{dataset_type}.beton",
        )

        data_loader = dataset.get_ffcv_loader(
            write_path=write_path,
            batch_size=batch_size,
            num_workers=loader_num_workers,
            device=device,
        )

    else:
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=loader_num_workers,
            pin_memory=loader_pin_memory,
            sampler=sampler,
        )

    if max_samples is not None:
        data_loader = early_stop_data_loader(
            data_loader, max_samples if max_samples > 1 else 1
        )

    return dataset, data_loader


# Model creation Helpers
def create_model(
    checkpoint_path: str,
    num_classes: int,
    recipe_path: Optional[str] = None,
    arch_key: Optional[str] = None,
    pretrained: Union[bool, str] = False,
    pretrained_dataset: Optional[str] = None,
    local_rank: int = -1,
    **model_kwargs,
) -> Tuple[Module, str]:
    """
    :param checkpoint_path: Path to the checkpoint to load. `zoo` for
        downloading weights with respect to a SparseZoo recipe
    :param num_classes: Integer representing the number of output classes
    :param recipe_path: Path or SparseZoo stub to the recipe for downloading,
        respective model. Defaults to `None`
    :param arch_key: The architecture key of the image classification model.
        Defaults to `None`
    :param pretrained: Whether to use pretrained weights or not. Defaults to
        False
    :param pretrained_dataset: The dataset to used for pretraining. Defaults to
        None
    :param local_rank: The local rank of the process. Defaults to -1
    :param model_kwargs: Additional keyword arguments to pass to the model
    :returns: A tuple containing the model and the model's arch_key
    """
    with torch_distributed_zero_first(local_rank):
        # only download once locally
        if checkpoint_path and checkpoint_path.lower() == "zoo":
            checkpoint_path = _download_model_from_zoo_using_recipe(
                recipe_stub=recipe_path,
            )

        result = ModelRegistry.create(
            key=arch_key,
            pretrained=pretrained,
            pretrained_path=checkpoint_path,
            pretrained_dataset=pretrained_dataset,
            num_classes=num_classes,
            **model_kwargs,
        )

        if not isinstance(result, tuple):
            model, arch_key = result, arch_key
        else:
            model, arch_key = result

        return model, arch_key


def infer_num_classes(
    train_dataset: Optional[Dataset],
    val_dataset: Optional[Dataset],
    dataset: str,
    model_kwargs: Dict[str, Any],
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
    :param checkpoint_path: Optional[str] The path to the checkpoint
    :return: str The arch_key to use for the model if present in the
        checkpoint
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
    checkpoint_path: Optional[str] = None
):
    """
    :param recipe_manager: The ScheduleModified manager to save recipes
    :param save_dir: The directory to save the recipe
    """
    recipe_save_path = os.path.join(save_dir, "recipe.yaml")
    if checkpoint_path:
        checkpoint_recipe = torch.load(checkpoint_path)['recipe']
        composed_manager = ScheduledModifierManager.compose_staged(base_recipe=deepcopy(checkpoint_recipe),
                                                                   additional_recipe=recipe_manager)
        composed_manager.save(recipe_save_path)

    else:
        recipe_manager.save(recipe_save_path)
    print(f"Saved recipe to {recipe_save_path}")


def save_model_training(
    model: Module,
    optim: Optimizer,
    manager,
    checkpoint_manager,
    save_name: str,
    save_dir: str,
    epoch: int,
    val_res: Optional[ModuleRunResults],
    arch_key: Optional[str] = None,
):
    """
    :param model: model architecture
    :param optim: the optimizer used
    :param recipe: the recipe used to obtain the model
    :param save_name: name to save model to
    :param save_dir: directory to save results in
    :param epoch: integer representing umber of epochs to
    :param val_res: results from validation run
    :param arch_key: if provided, the `arch_key` will be saved in the
        checkpoint
    """
    if checkpoint_manager:
        recipe = str(ScheduledModifierManager.compose_staged(base_recipe=checkpoint_manager,
                                                             additional_recipe=manager))
    else:
        recipe = str(manager)

    has_top1 = "top1acc" in val_res.results
    metric_name = "top-1 accuracy" if has_top1 else "val_loss"
    metric = val_res.result_mean("top1acc" if has_top1 else DEFAULT_LOSS_KEY).item()
    print(
        f"Saving model for epoch {epoch} and {metric_name} "
        f"{metric} to {save_dir} for {save_name}"
    )
    exporter = ModuleExporter(model, save_dir)
    exporter.export_pytorch(
        optimizer=optim,
        epoch=epoch,
        recipe=recipe,
        name = f"{save_name}.pth",
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


# TODO: Add type for training_args
def extract_metadata(metadata_args: List[str], training_args)-> Dict[str, Any]:
    """
    Extract metadata from the training arguments.

    :param metadata_args: List of keys we are attempting to retrieve from `training_args`
        and pass as metadata
    :param training_args: TrainingArguments of the pipeline
    :return: metadata
    """
    # TODO: Possibly share this functionality among IC and transformers (and future pipelines)
    metadata = {}
    training_args_dict = asdict(training_args)

    for arg in metadata_args:
        if arg not in training_args_dict.keys():
            logging.warning(
                f"Required metadata argument {arg} was not found "
                f"in the training arguments. Setting {arg} to None."
            )
            metadata[arg] = None
        else:
            metadata[arg] = training_args_dict[arg]

    return metadata


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


@contextmanager
def _nullcontext(enter_result=None):
    """
    A context manager that does nothing. For compatibility with Python 3.6
    Update usages to use contextlib.nullcontext on Python 3.7+
    """
    yield enter_result
