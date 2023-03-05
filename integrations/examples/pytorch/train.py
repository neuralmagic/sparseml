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
######
Command help:
usage: train.py [-h] --train-batch-size TRAIN_BATCH_SIZE --test-batch-size
                TEST_BATCH_SIZE --arch-key ARCH_KEY --dataset DATASET
                --dataset-path DATASET_PATH
                [--checkpoint-path CHECKPOINT_PATH] [--init-lr INIT_LR]
                [--optim-args OPTIM_ARGS] [--recipe-path RECIPE_PATH]
                [--sparse-transfer-learn [SPARSE_TRANSFER_LEARN]]
                [--eval-mode [EVAL_MODE]] [--optim OPTIM]
                [--logs-dir LOGS_DIR] [--save-best-after SAVE_BEST_AFTER]
                [--save-epochs SAVE_EPOCHS]
                [--use-mixed-precision [USE_MIXED_PRECISION]]
                [--debug-steps DEBUG_STEPS] [--pretrained PRETRAINED]
                [--pretrained-dataset PRETRAINED_DATASET]
                [--model-kwargs MODEL_KWARGS]
                [--dataset-kwargs DATASET_KWARGS] [--model-tag MODEL_TAG]
                [--save-dir SAVE_DIR] [--device DEVICE]
                [--loader-num-workers LOADER_NUM_WORKERS]
                [--no-loader-pin-memory]
                [--loader-pin-memory [LOADER_PIN_MEMORY]]

optional arguments:
  -h, --help            show this help message and exit
  --train-batch-size TRAIN_BATCH_SIZE
                        The batch size to use while training
  --test-batch-size TEST_BATCH_SIZE
                        The batch size to use while testing
  --arch-key ARCH_KEY   The type of model to use, ex: resnet50, vgg16,
                        mobilenet put as help to see the full list (will raise
                        an exception with the list)
  --dataset DATASET     The dataset to use for training, ex: imagenet,
                        imagenette, cifar10, etc. Set to imagefolder for a
                        generic dataset setup with an image folder structure
                        setup like imagenet or loadable by a dataset in
                        sparseml.pytorch.datasets
  --dataset-path DATASET_PATH
                        The root path to where the dataset is stored
  --checkpoint-path CHECKPOINT_PATH
                        A path to a previous checkpoint to load the state from
                        and resume the state for. If provided, pretrained will
                        be ignored. If using a SparseZoo recipe, can also
                        provide 'zoo' to load the base weights associated with
                        that recipe
  --init-lr INIT_LR     The initial learning rate to use while training, the
                        actual initial value used should be set by the
                        sparseml recipe
  --optim-args OPTIM_ARGS
                        Additional args to be passed to the optimizer passed
                        in as a json object
  --recipe-path RECIPE_PATH
                        The path to the yaml file containing the modifiers and
                        schedule to apply them with. Can also provide a
                        SparseZoo stub prefixed with 'zoo:' with an optional
                        '?recipe_type=' argument
  --sparse-transfer-learn [SPARSE_TRANSFER_LEARN]
                        Enable sparse transfer learning modifiers to enforce
                        the sparsity for already sparse layers. The modifiers
                        are added to the ones to be loaded from the recipe-
                        path
  --eval-mode [EVAL_MODE]
                        Puts into evaluation mode so that the model can be
                        evaluated on the desired dataset
  --optim OPTIM         The optimizer type to use, one of [SGD, Adam, RMSprop]
  --logs-dir LOGS_DIR   The path to the directory for saving logs
  --save-best-after SAVE_BEST_AFTER
                        start saving the best validation result after the
                        given epoch completes until the end of training
  --save-epochs SAVE_EPOCHS
                        epochs to save checkpoints at
  --use-mixed-precision [USE_MIXED_PRECISION]
                        Trains model using mixed precision. Supported
                        environments are single GPU and multiple GPUs using
                        DistributedDataParallel with one GPU per process
  --debug-steps DEBUG_STEPS
                        Amount of steps to run for training and testing for a
                        debug mode
  --pretrained PRETRAINED
                        The type of pretrained weights to use, default is true
                        to load the default pretrained weights for the model.
                        Otherwise should be set to the desired weights type:
                        [base, optim, optim-perf]. To not load any weights set
                        to one of [none, false]
  --pretrained-dataset PRETRAINED_DATASET
                        The dataset to load pretrained weights for if
                        pretrained is set. Default is None which will load the
                        default dataset for the architecture. Ex can be set to
                        imagenet, cifar10, etc
  --model-kwargs MODEL_KWARGS
                        Keyword arguments to be passed to model constructor,
                        should be given as a json object
  --dataset-kwargs DATASET_KWARGS
                        Keyword arguments to be passed to dataset constructor,
                        should be given as a json object
  --model-tag MODEL_TAG
                        A tag to use for the model for saving results under
                        save-dir, defaults to the model arch and dataset used
  --save-dir SAVE_DIR   The path to the directory for saving results
  --device DEVICE       The device to run on (can also include ids for data
                        parallel), ex: cpu, cuda, cuda:0,1
  --loader-num-workers LOADER_NUM_WORKERS
                        The number of workers to use for data loading
  --no-loader-pin-memory
                        Do not use pinned memory for data loading
  --loader-pin-memory [LOADER_PIN_MEMORY]
                        Use pinned memory for data loading
#########
EXAMPLES
#########

##########
Example command for pruning resnet50 on imagenet dataset:
python integrations/pytorch/train.py \
    --recipe-path ~/sparseml_recipes/pruning_resnet50.yaml \
    --arch-key resnet50 --dataset imagenet --dataset-path ~/datasets/ILSVRC2012 \
    --train-batch-size 256 --test-batch-size 1024

##########
Example command for transfer learning sparse mobilenet_v1 on an image folder dataset:
python integrations/pytorch/train.py \
    --sparse-transfer-learn \
    --recipe-path  ~/sparseml_recipes/pruning_mobilenet.yaml \
    --arch-key mobilenet_v1 --pretrained pruned-moderate \
    --dataset imagefolder --dataset-path ~/datasets/my_imagefolder_dataset \
    --train-batch-size 256 --test-batch-size 1024

##########
Template command for running training with this script on multiple GPUs using
DistributedDataParallel using mixed precision. Note - DDP support in this script
only tested for torch==1.7.0.
python -m torch.distributed.launch \
--nproc_per_node <NUM GPUs> \
integrations/pytorch/train.py \
--use-mixed-precision \
<TRAIN.PY ARGUMENTS>
"""
import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

import utils
from argparser_.nm_argparser_ import NmArgumentParser
from sparseml import get_main_logger
from sparseml.pytorch.models import ModelRegistry
from sparseml.pytorch.utils import (
    DEFAULT_LOSS_KEY,
    ModuleDeviceContext,
    ModuleTester,
    ModuleTrainer,
    default_device,
    get_prunable_layers,
    model_to_device,
    set_deterministic_seeds,
    tensor_sparsity,
)


CURRENT_TASK = utils.Tasks.TRAIN
LOGGER = get_main_logger()


@dataclass
class TrainingArguments:
    """
    Represents the arguments we use in our PyTorch integration scripts for
    training tasks

    Using :class:`NmArgumentParser` we can turn this class into `argparse
    <https://docs.python.org/3/library/argparse.html#module-argparse>`__
    arguments that can be specified on the command line.

    :param train_batch_size: An int representing the training batch size.
    :param test_batch_size: An int representing the test batch size.
    :param arch_key: A str key representing the type of model to use,
        ex:resnet50.
    :param dataset: The dataset to use for training, ex imagenet, imagenette,
        etc; Set to `imagefolder` for a custom dataset.
    :param dataset_path: Root path to dataset location.
    :param local_rank: DDP argument set by PyTorch in DDP mode, default -1
    :param checkpoint_path: A path to a previous checkpoint to load the state
        from and resume the state for; Also works with SparseZoo recipes;
        Set to zoo to automatically download and load weights associated with a
        recipe.
    :param init_lr: float representing the initial learning for training,
        default=1e-9 .
    :param optim_args: Additional arguments to be passed in to the optimizer as
        a json object
    :param recipe_path: The path to the yaml file containing the modifiers and
        schedule to apply them with; Can also provide a SparseZoo stub prefixed
        with 'zoo:'.
    :param sparse_transfer_learn: Boolean to enable sparse transfer learning
        modifiers to enforce
        the sparsity for already sparse layers. The modifiers are added to
        the ones to be loaded from the recipe-path.
    :param eval_mode: bool to start evaluation mode so that the model can be
        evaluated on the desired dataset.
    :param optim: str respresnting the optimizer type to use, one of
        [SGD, Adam, RMSprop].
    :param logs_dir: The path to the directory for saving logs.
    :param save_best_after: int epoch number to start saving the best
        validation result after until the end of training.
    :param save_epochs: int epochs to save checkpoints at.
    :param use_mixed_precision: bool to train model using mixed precision.
        Supported environments are single GPU and multiple GPUs using
        DistributedDataParallel with one GPU per process.
    :param debug_steps: int represnting amount of steps to run for training and
        testing for debug mode default=-1.
    :param pretrained: The type of pretrained weights to use default is true
        to load the default pretrained weights for the model Otherwise should
        be set to the desired weights type: [base, optim, optim-perf];
        To not load any weights set to one of [none, false].
    :param pretrained_dataset: str representing the dataset to load pretrained
        weights for if pretrained is set; Default is None which will load the
        default dataset for the architecture; Ex can be set to imagenet,
        cifar10, etc".
    :param model_kwargs: json object containing keyword arguments to be
        passed to model constructor.
    :param dataset_kwargs: json object to load keyword arguments to be passed
        to dataset constructor.
    :param model_tag: A str tag to use for the model for saving results
        under save-dir, defaults to the model arch and dataset used.
    :param save_dir: The path to the directory for saving results,
        default="pytorch_vision".
    :param device: str represnting the device to run on (can also include ids
        for data parallel), ex:{cpu, cuda, cuda:0,1}.
    :param loader_num_workers: int number of workers to use for data loading,
        default=4.
    :param loader_pin_memory: bool to use pinned memory for data loading,
        default=True.
    """

    train_batch_size: int = field(
        metadata={"help": "The batch size to use while training"}
    )

    test_batch_size: int = field(
        metadata={"help": "The batch size to use while testing"}
    )
    arch_key: str = field(
        metadata={
            "help": "The type of model to use, ex: resnet50, vgg16, mobilenet "
            "put as help to see the full list (will raise an exception"
            "with the list)",
        }
    )

    dataset: str = field(
        metadata={
            "help": "The dataset to use for training, "
            "ex: imagenet, imagenette, cifar10, etc. "
            "Set to imagefolder for a generic dataset setup "
            "with an image folder structure setup like imagenet or"
            " loadable by a dataset in sparseml.pytorch.datasets"
        }
    )

    dataset_path: str = field(
        metadata={
            "help": "The root path to where the dataset is stored",
        }
    )
    local_rank: int = field(
        default=-1,
        metadata={
            "keep_underscores": True,
            "help": argparse.SUPPRESS,
        },
    )

    checkpoint_path: str = field(
        default=None,
        metadata={
            "help": "A path to a previous checkpoint to load the state from "
            "and resume the state for. If provided, pretrained will "
            "be ignored . If using a SparseZoo recipe, can also "
            "provide 'zoo' to load the base weights associated with "
            "that recipe"
        },
    )

    init_lr: float = field(
        default=1e-9,
        metadata={
            "help": "The initial learning rate to use while training, "
            "the actual initial value used should be set by the"
            " sparseml recipe"
        },
    )

    optim_args: json.loads = field(
        default_factory=lambda: {
            "momentum": 0.9,
            "nesterov": True,
            "weight_decay": 0.0001,
        },
        metadata={
            "help": "Additional args to be passed to the optimizer passed in"
            " as a json object",
        },
    )

    recipe_path: str = field(
        default=None,
        metadata={
            "help": "The path to the yaml file containing the modifiers and "
            "schedule to apply them with. Can also provide a "
            "SparseZoo stub prefixed with 'zoo:' with an optional "
            "'?recipe_type=' argument"
        },
    )

    sparse_transfer_learn: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enable sparse transfer learning modifiers to enforce the "
            "sparsity for already sparse layers. The modifiers are "
            "added to the ones to be loaded from the recipe-path"
        },
    )

    eval_mode: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Puts into evaluation mode so that the model can be "
            "evaluated on the desired dataset"
        },
    )

    optim: str = field(
        default="SGD",
        metadata={"help": "The optimizer type to use, one of [SGD, Adam, RMSprop]"},
    )

    logs_dir: str = field(
        default=os.path.join("pytorch_vision_train", "tensorboard-logs"),
        metadata={
            "help": "The path to the directory for saving logs",
        },
    )

    save_best_after: int = field(
        default=-1,
        metadata={
            "help": "start saving the best validation result after the given "
            "epoch completes until the end of training"
        },
    )
    save_epochs: List[int] = field(
        default_factory=lambda: [], metadata={"help": "epochs to save checkpoints at"}
    )

    use_mixed_precision: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Trains model using mixed precision. Supported "
            "environments are single GPU and multiple GPUs using "
            "DistributedDataParallel with one GPU per process"
        },
    )

    debug_steps: int = field(
        default=-1,
        metadata={
            "help": "Amount of steps to run for training and testing for a "
            "debug mode"
        },
    )

    pretrained: str = field(
        default=True,
        metadata={
            "help": "The type of pretrained weights to use, "
            "default is true to load the default pretrained weights for "
            "the model. Otherwise should be set to the desired weights "
            "type: [base, optim, optim-perf]. To not load any weights set"
            "to one of [none, false]"
        },
    )

    pretrained_dataset: str = field(
        default=None,
        metadata={
            "help": "The dataset to load pretrained weights for if pretrained is "
            "set. Default is None which will load the default dataset for "
            "the architecture. Ex can be set to imagenet, cifar10, etc",
        },
    )

    model_kwargs: json.loads = field(
        default_factory=lambda: {},
        metadata={
            "help": "Keyword arguments to be passed to model constructor, should "
            "be given as a json object"
        },
    )

    dataset_kwargs: json.loads = field(
        default_factory=lambda: {},
        metadata={
            "help": "Keyword arguments to be passed to dataset constructor, "
            "should be given as a json object",
        },
    )

    model_tag: str = field(
        default=None,
        metadata={
            "help": "A tag to use for the model for saving results under save-dir, "
            "defaults to the model arch and dataset used",
        },
    )

    save_dir: str = field(
        default="pytorch_vision",
        metadata={
            "help": "The path to the directory for saving results",
        },
    )

    device: str = field(
        default=default_device(),
        metadata={
            "help": "The device to run on (can also include ids for data "
            "parallel), ex: cpu, cuda, cuda:0,1"
        },
    )

    loader_num_workers: int = field(
        default=4, metadata={"help": "The number of workers to use for data loading"}
    )

    loader_pin_memory: bool = field(
        default=True, metadata={"help": "Use pinned memory for data loading"}
    )

    def __post_init__(self):
        # add ddp args
        env_world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.world_size = env_world_size

        env_rank = int(os.environ.get("RANK", -1))
        self.rank = env_rank

        self.is_main_process = self.rank in [
            -1,
            0,
        ]  # non DDP execution or 0th DDP process

        # modify training batch size for give world size
        assert self.train_batch_size % self.world_size == 0, (
            f"Invalid training batch size for world size {self.world_size} "
            f"given batch size {self.train_batch_size}. "
            f"world size must divide training batch size evenly."
        )

        self.train_batch_size = self.train_batch_size // self.world_size

        if "preprocessing_type" not in self.dataset_kwargs and (
            "coco" in self.dataset.lower() or "voc" in self.dataset.lower()
        ):
            if "ssd" in self.arch_key.lower():
                self.dataset_kwargs["preprocessing_type"] = "ssd"
            elif "yolo" in self.arch_key.lower():
                self.dataset_kwargs["preprocessing_type"] = "yolo"

        if self.local_rank != -1:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            set_deterministic_seeds(0)

        self.approximate = False


def train(
    train_args: TrainingArguments,
    model: Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_shape: Tuple[int, ...],
    save_dir: str,
    loggers: List[Any],
) -> None:
    """
    Utility function to drive the training processing

    :param train_args: A TrainingArguments object with
        arguments for current training task
    :param model: model architecture to train
    :param train_loader: A DataLoader for training data
    :param val_loader: A DataLoader for validation data
    :param input_shape: A tuple of integers representing the shape of inputs
    :param save_dir: Directory to store checkpoints at during training process
    :param loggers: List of loggers to use during training process
    """
    # loss setup
    val_loss = utils.get_loss_wrapper(arch_key=train_args.arch_key, training=True)
    LOGGER.info(f"created loss for validation: {val_loss}")

    train_loss = utils.get_loss_wrapper(arch_key=train_args.arch_key, training=True)
    LOGGER.info(f"created loss for training: {train_loss}")

    # training setup
    if not train_args.eval_mode:
        epoch, optim, manager = utils.create_scheduled_optimizer(
            train_args,
            model,
            train_loader,
            loggers,
        )
    else:
        epoch = 0
        train_loss = None
        optim = None
        manager = None

    # device setup
    if train_args.rank == -1:
        device = train_args.device
        ddp = False
    else:
        torch.cuda.set_device(train_args.local_rank)
        device = train_args.local_rank
        ddp = True

    model, device, device_ids = model_to_device(model, device, ddp=ddp)
    LOGGER.info(f"running on device {device} for ids {device_ids}")

    trainer = (
        ModuleTrainer(
            model,
            device,
            train_loss,
            optim,
            loggers=loggers,
            device_context=ModuleDeviceContext(
                use_mixed_precision=train_args.use_mixed_precision,
                world_size=train_args.world_size,
            ),
        )
        if not train_args.eval_mode
        else None
    )

    if train_args.is_main_process:  # only test on one DDP process if using DDP
        tester = ModuleTester(model, device, val_loss, loggers=loggers, log_steps=-1)

        # initial baseline eval run
        tester.run_epoch(
            val_loader, epoch=epoch - 1, max_epochs=-1, max_steps=train_args.debug_steps
        )

    if not train_args.eval_mode:
        utils.save_recipe(recipe_manager=manager, save_dir=save_dir)
        LOGGER.info(f"starting training from epoch {epoch}")

        if epoch > 0:
            LOGGER.info("adjusting ScheduledOptimizer to restore point")
            optim.adjust_current_step(epoch, 0)

        target_metric = (
            "top1acc" if "top1acc" in tester.loss.available_losses else DEFAULT_LOSS_KEY
        )
        best_metric = None
        val_res = None

        while epoch < manager.max_epochs:
            if train_args.debug_steps > 0:
                # correct since all optimizer steps are not
                # taken in the epochs for debug mode
                optim.adjust_current_step(epoch, 0)

            if train_args.rank != -1:  # sync DDP dataloaders
                train_loader.sampler.set_epoch(epoch)

            trainer.run_epoch(
                train_loader,
                epoch,
                max_epochs=manager.max_epochs,
                max_steps=train_args.debug_steps,
                show_progress=train_args.is_main_process,
            )

            # testing steps
            if train_args.is_main_process:
                # only test and save on main process
                val_res = tester.run_epoch(
                    val_loader,
                    epoch,
                    max_epochs=manager.max_epochs,
                    max_steps=train_args.debug_steps,
                )
                val_metric = val_res.result_mean(target_metric).item()

                if epoch >= train_args.save_best_after and (
                    best_metric is None
                    or (
                        val_metric <= best_metric
                        if target_metric != "top1acc"
                        else val_metric >= best_metric
                    )
                ):
                    utils.save_model_training(
                        model,
                        optim,
                        input_shape,
                        "checkpoint-best",
                        save_dir,
                        epoch,
                        val_res,
                    )
                    best_metric = val_metric

            # save checkpoints
            _save_epoch = (
                train_args.is_main_process
                and train_args.save_epochs
                and epoch in train_args.save_epochs
            )
            if _save_epoch:
                utils.save_model_training(
                    model,
                    optim,
                    input_shape,
                    f"checkpoint-{epoch:04d}-{val_metric:.04f}",
                    save_dir,
                    epoch,
                    val_res,
                )

            epoch += 1

        # export the final model
        LOGGER.info("completed...")
        if train_args.is_main_process:
            # only convert qat -> quantized ONNX graph for finalized model
            # TODO: change this to all checkpoints when conversion times improve
            utils.save_model_training(
                model, optim, input_shape, "model", save_dir, epoch - 1, val_res, True
            )

            LOGGER.info("layer sparsities:")
            for (name, layer) in get_prunable_layers(model):
                LOGGER.info(
                    f"{name}.weight: {tensor_sparsity(layer.weight).item():.4f}"
                )

    # close DDP
    if train_args.rank != -1:
        torch.distributed.destroy_process_group()


def main():
    """
    Driver function for the script
    """
    _parser = NmArgumentParser(dataclass_types=TrainingArguments)
    training_args, _ = _parser.parse_args_into_dataclasses()

    save_dir, loggers = utils.get_save_dir_and_loggers(training_args, task=CURRENT_TASK)

    input_shape = ModelRegistry.input_shape(training_args.arch_key)
    image_size = input_shape[1]  # assume shape [C, S, S] where S is the image size

    (
        train_dataset,
        train_loader,
        val_dataset,
        val_loader,
    ) = utils.get_train_and_validation_loaders(
        training_args, image_size, task=CURRENT_TASK
    )

    num_classes = utils.infer_num_classes(training_args, train_dataset, val_dataset)

    # # model creation
    model = utils.create_model(training_args, num_classes)
    train(
        training_args, model, train_loader, val_loader, input_shape, save_dir, loggers
    )


if __name__ == "__main__":
    main()
