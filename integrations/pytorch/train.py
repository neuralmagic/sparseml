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
usage: train.py [-h] --arch-key ARCH_KEY [--pretrained PRETRAINED]
                [--pretrained-dataset PRETRAINED_DATASET]
                [--model-kwargs MODEL_KWARGS] --dataset DATASET --dataset-path
                DATASET_PATH [--dataset-kwargs DATASET_KWARGS]
                [--model-tag MODEL_TAG] [--save-dir SAVE_DIR]
                [--device DEVICE] [--loader-num-workers LOADER_NUM_WORKERS]
                [--loader-pin-memory LOADER_PIN_MEMORY]
                [--checkpoint-path CHECKPOINT_PATH] [--init-lr INIT_LR]
                [--optim-args OPTIM_ARGS] [--recipe-path RECIPE_PATH]
                [--sparse-transfer-learn] [--eval-mode] --train-batch-size
                TRAIN_BATCH_SIZE --test-batch-size TEST_BATCH_SIZE
                [--optim OPTIM] [--logs-dir LOGS_DIR]
                [--save-best-after SAVE_BEST_AFTER]
                [--save-epochs SAVE_EPOCHS [SAVE_EPOCHS ...]]
                [--use-mixed-precision] [--debug-steps DEBUG_STEPS]

Train and/or prune an image classification model on a dataset

optional arguments:
  -h, --help            show this help message and exit
  --arch-key ARCH_KEY   The type of model to use, ex: resnet50, vgg16,
                        mobilenet put as help to see the full list (will raise
                        an exception with the list)
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
  --dataset DATASET     The dataset to use for training, ex: imagenet,
                        imagenette, cifar10, etc. Set to imagefolder for a
                        generic dataset setup with an image folder structure
                        setup like imagenet or loadable by a dataset in
                        sparseml.pytorch.datasets
  --dataset-path DATASET_PATH
                        The root path to where the dataset is stored
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
  --loader-pin-memory LOADER_PIN_MEMORY
                        Use pinned memory for data loading
  --checkpoint-path CHECKPOINT_PATH
                        A path to a previous checkpoint to load the state from
                        and resume the state for. If provided, pretrained will
                        be ignored. If using a SparseZoo recipe, can also
                        provide 'zoo' to load the base weights associated with
                        that recipe. If using a SparseZoo recipe, can also
                        provide 'zoo' to load the base weights associated with
                        that recipe
  --init-lr INIT_LR     The initial learning rate to use for the sensitivity
                        analysisThe initial learning rate to use while
                        training, the actual initial value used should be set
                        by the sparseml recipe
  --optim-args OPTIM_ARGS
                        Additional args to be passed to the optimizer passed
                        in as a json object
  --recipe-path RECIPE_PATH
                        The path to the yaml file containing the modifiers and
                        schedule to apply them with. Can also provide a
                        SparseZoo stub prefixed with 'zoo:' with an optional
                        '?recipe_type=' argument
  --sparse-transfer-learn
                        Enable sparse transfer learning modifiers to enforce
                        the sparsity for already sparse layers. The modifiers
                        are added to the ones to be loaded from the recipe-
                        path
  --eval-mode           Puts into evaluation mode so that the model can be
                        evaluated on the desired dataset
  --train-batch-size TRAIN_BATCH_SIZE
                        The batch size to use while training
  --test-batch-size TEST_BATCH_SIZE
                        The batch size to use while testing
  --optim OPTIM         The optimizer type to use, one of [SGD, Adam]
  --logs-dir LOGS_DIR   The path to the directory for saving logs
  --save-best-after SAVE_BEST_AFTER
                        start saving the best validation result after the
                        given epoch completes until the end of training
  --save-epochs SAVE_EPOCHS [SAVE_EPOCHS ...]
                        epochs to save checkpoints at
  --use-mixed-precision
                        Trains model using mixed precision. Supported
                        environments are single GPU and multiple GPUs using
                        DistributedDataParallel with one GPU per process
  --debug-steps DEBUG_STEPS
                        Amount of steps to run for training and testing for a
                        debug mode

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
from typing import Any, List, Tuple

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

import utils
from sparseml import get_main_logger
from sparseml.pytorch.models import ModelRegistry
from sparseml.pytorch.utils import (
    DEFAULT_LOSS_KEY,
    ModuleDeviceContext,
    ModuleTester,
    ModuleTrainer,
    get_prunable_layers,
    model_to_device,
    tensor_sparsity,
)


CURRENT_TASK = utils.Tasks.TRAIN
LOGGER = get_main_logger()


def train(
    args: argparse.Namespace,
    model: Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_shape: Tuple[int, ...],
    save_dir: str,
    loggers: List[Any],
) -> None:
    """
    Utility function to drive the training processing

    :params args : A Namespace object containing at-least the following keys
        arch_key, eval_mode, rank, use_mixed_precision, world_size, is_main_process
    :param model: model architecture to train
    :param train_loader: A DataLoader for training data
    :param val_loader: A DataLoader for validation data
    :param input_shape: A tuple of integers representing the shape of inputs
    :param save_dir: Directory to store checkpoints at during training process
    :param loggers: List of loggers to use during training process
    """
    # loss setup
    val_loss = utils.get_loss_wrapper(args, training=True)
    LOGGER.info("created loss for validation: {}".format(val_loss))

    train_loss = utils.get_loss_wrapper(args, training=True)
    LOGGER.info("created loss for training: {}".format(train_loss))

    # training setup
    if not args.eval_mode:
        epoch, optim, manager = utils.create_scheduled_optimizer(
            args,
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
    if args.rank == -1:
        device = args.device
        ddp = False
    else:
        torch.cuda.set_device(args.local_rank)
        device = args.local_rank
        ddp = True

    model, device, device_ids = model_to_device(model, device, ddp=ddp)
    LOGGER.info("running on device {} for ids {}".format(device, device_ids))

    trainer = (
        ModuleTrainer(
            model,
            device,
            train_loss,
            optim,
            loggers=loggers,
            device_context=ModuleDeviceContext(
                use_mixed_precision=args.use_mixed_precision, world_size=args.world_size
            ),
        )
        if not args.eval_mode
        else None
    )

    if args.is_main_process:  # only test on one DDP process if using DDP
        tester = ModuleTester(model, device, val_loss, loggers=loggers, log_steps=-1)

        # initial baseline eval run
        tester.run_epoch(val_loader, epoch=epoch - 1, max_steps=args.debug_steps)

    if not args.eval_mode:
        utils.save_recipe(recipe_manager=manager, save_dir=save_dir)
        LOGGER.info("starting training from epoch {}".format(epoch))

        if epoch > 0:
            LOGGER.info("adjusting ScheduledOptimizer to restore point")
            optim.adjust_current_step(epoch, 0)

        best_loss = None
        val_res = None

        while epoch < manager.max_epochs:
            if args.debug_steps > 0:
                # correct since all optimizer steps are not
                # taken in the epochs for debug mode
                optim.adjust_current_step(epoch, 0)

            if args.rank != -1:  # sync DDP dataloaders
                train_loader.sampler.set_epoch(epoch)

            trainer.run_epoch(
                train_loader,
                epoch,
                max_steps=args.debug_steps,
                show_progress=args.is_main_process,
            )

            # testing steps
            if args.is_main_process:  # only test and save on main process
                val_res = tester.run_epoch(
                    val_loader, epoch, max_steps=args.debug_steps
                )
                val_loss = val_res.result_mean(DEFAULT_LOSS_KEY).item()

                if epoch >= args.save_best_after and (
                    best_loss is None or val_loss <= best_loss
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
                    best_loss = val_loss

            # save checkpoints
            if args.is_main_process and args.save_epochs and epoch in args.save_epochs:
                utils.save_model_training(
                    model,
                    optim,
                    input_shape,
                    "checkpoint-{:04d}-{:.04f}".format(epoch, val_loss),
                    save_dir,
                    epoch,
                    val_res,
                )

            epoch += 1

        # export the final model
        LOGGER.info("completed...")
        if args.is_main_process:
            # only convert qat -> quantized ONNX graph for finalized model
            # TODO: change this to all checkpoints when conversion times improve
            utils.save_model_training(
                model, optim, input_shape, "model", save_dir, epoch - 1, val_res, True
            )

            LOGGER.info("layer sparsities:")
            for (name, layer) in get_prunable_layers(model):
                LOGGER.info(
                    "{}.weight: {:.4f}".format(
                        name, tensor_sparsity(layer.weight).item()
                    )
                )

    # close DDP
    if args.rank != -1:
        torch.distributed.destroy_process_group()


def parse_args():
    """
    Utility method to parse command line arguments for Training specific tasks
    """
    parser = argparse.ArgumentParser(
        description="Train and/or prune an image classification model on a dataset"
    )

    utils.add_local_rank(parser)
    utils.add_universal_args(parser)
    utils.add_device_args(parser)
    utils.add_workers_args(parser)
    utils.add_pin_memory_args(parser)

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help=(
            "A path to a previous checkpoint to load the state from and "
            "resume the state for. If provided, pretrained will be ignored"
            ". If using a SparseZoo recipe, can also provide 'zoo' to load "
            "the base weights associated with that recipe"
            ". If using a SparseZoo recipe, can also provide 'zoo' to load "
            "the base weights associated with that recipe"
        ),
    )
    parser.add_argument(
        "--init-lr",
        type=float,
        default=1e-9,
        help=(
            "The initial learning rate to use for the sensitivity analysis"
            "The initial learning rate to use while training, "
            "the actual initial value used should be set by the sparseml recipe"
        ),
    )
    parser.add_argument(
        "--optim-args",
        type=json.loads,
        default=({"momentum": 0.9, "nesterov": True, "weight_decay": 0.0001}),
        help="Additional args to be passed to the optimizer passed in"
        " as a json object",
    )
    parser.add_argument(
        "--recipe-path",
        type=str,
        default=None,
        help="The path to the yaml file containing the modifiers and "
        "schedule to apply them with. Can also provide a SparseZoo stub "
        "prefixed with 'zoo:' with an optional '?recipe_type=' argument",
    )
    parser.add_argument(
        "--sparse-transfer-learn",
        action="store_true",
        help=(
            "Enable sparse transfer learning modifiers to enforce the sparsity "
            "for already sparse layers. The modifiers are added to the "
            "ones to be loaded from the recipe-path"
        ),
    )
    parser.add_argument(
        "--eval-mode",
        action="store_true",
        help="Puts into evaluation mode so that the model can be "
        "evaluated on the desired dataset",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        required=True,
        help="The batch size to use while training",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        required=True,
        help="The batch size to use while testing",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="SGD",
        help="The optimizer type to use, one of [SGD, Adam]",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default=os.path.join("pytorch_vision_train", "tensorboard-logs"),
        help="The path to the directory for saving logs",
    )
    parser.add_argument(
        "--save-best-after",
        type=int,
        default=-1,
        help="start saving the best validation result after the given "
        "epoch completes until the end of training",
    )
    parser.add_argument(
        "--save-epochs",
        type=int,
        default=[],
        nargs="+",
        help="epochs to save checkpoints at",
    )
    parser.add_argument(
        "--use-mixed-precision",
        action="store_true",
        help=(
            "Trains model using mixed precision. Supported environments are "
            "single GPU and multiple GPUs using DistributedDataParallel with "
            "one GPU per process"
        ),
    )
    parser.add_argument(
        "--debug-steps",
        type=int,
        default=-1,
        help="Amount of steps to run for training and testing for a debug mode",
    )

    args = parser.parse_args()
    args.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.rank = int(os.environ["RANK"]) if "RANK" in os.environ else -1
    args.is_main_process = args.rank in [-1, 0]  # non DDP execution or 0th DDP process

    # modify training batch size for give world size
    assert args.train_batch_size % args.world_size == 0, (
        f"Invalid training batch size for world size {args.world_size} "
        f"given batch size {args.train_batch_size}. "
        f"world size must divide training batch size evenly."
    )

    args.train_batch_size = args.train_batch_size // args.world_size
    utils.append_preprocessing_args(args)
    return args


def main():
    """
    Driver function for the script
    """
    args_ = parse_args()
    utils.distributed_setup(args_.local_rank)
    save_dir, loggers = utils.get_save_dir_and_loggers(args_, task=CURRENT_TASK)

    input_shape = ModelRegistry.input_shape(args_.arch_key)
    image_size = input_shape[1]  # assume shape [C, S, S] where S is the image size

    (
        train_dataset,
        train_loader,
        val_dataset,
        val_loader,
    ) = utils.get_train_and_validation_loaders(args_, image_size, task=CURRENT_TASK)

    # model creation
    num_classes = utils.infer_num_classes(args_, train_dataset, val_dataset)

    model = utils.create_model(args_, num_classes)

    train(args_, model, train_loader, val_loader, input_shape, save_dir, loggers)


if __name__ == "__main__":
    main()
