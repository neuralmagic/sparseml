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
#####
Command help:
usage: analysis.py [-h] {pr_sensitivity,lr_sensitivity} ...

Utility script to Run a kernel sparsity (pruning) or learning rate sensitivity
analysis for a desired image classification architecture

positional arguments:
  {pr_sensitivity,lr_sensitivity}

optional arguments:
  -h, --help            show this help message and exit

######
pr-sensitivity command help:
usage: analysis.py pr_sensitivity [-h] --arch-key ARCH_KEY
                                  [--pretrained PRETRAINED]
                                  [--pretrained-dataset PRETRAINED_DATASET]
                                  [--model-kwargs MODEL_KWARGS] --dataset
                                  DATASET --dataset-path DATASET_PATH
                                  [--dataset-kwargs DATASET_KWARGS]
                                  [--model-tag MODEL_TAG]
                                  [--save-dir SAVE_DIR] [--device DEVICE]
                                  [--loader-num-workers LOADER_NUM_WORKERS]
                                  [--loader-pin-memory LOADER_PIN_MEMORY]
                                  [--checkpoint-path CHECKPOINT_PATH]
                                  [--steps-per-measurement STEPS_PER_MEASUREMENT]
                                  [--batch-size BATCH_SIZE] [--approximate]

Run a kernel sparsity (pruning) analysis for a given model

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
                        that recipe
  --steps-per-measurement STEPS_PER_MEASUREMENT
                        The number of steps (batches) to run for each
                        measurement
  --batch-size BATCH_SIZE
                        The batch size to use while training
  --approximate         True to approximate without running data through the
                        model, otherwise will run a one shot analysis

######
lr_sensitivity command help:

usage: analysis.py lr_sensitivity [-h] --arch-key ARCH_KEY
                                  [--pretrained PRETRAINED]
                                  [--pretrained-dataset PRETRAINED_DATASET]
                                  [--model-kwargs MODEL_KWARGS] --dataset
                                  DATASET --dataset-path DATASET_PATH
                                  [--dataset-kwargs DATASET_KWARGS]
                                  [--model-tag MODEL_TAG]
                                  [--save-dir SAVE_DIR] [--device DEVICE]
                                  [--loader-num-workers LOADER_NUM_WORKERS]
                                  [--loader-pin-memory LOADER_PIN_MEMORY]
                                  [--checkpoint-path CHECKPOINT_PATH]
                                  [--init-lr INIT_LR]
                                  [--optim-args OPTIM_ARGS]
                                  [--final-lr FINAL_LR]
                                  [--steps-per-measurement STEPS_PER_MEASUREMENT]
                                  --batch-size BATCH_SIZE

Run a learning rate sensitivity analysis for a desired image classification or
detection architecture

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
                        that recipe
  --init-lr INIT_LR     The initial learning rate to use for the sensitivity
                        analysis
  --optim-args OPTIM_ARGS
                        Additional args to be passed to the optimizer passed
                        in as a json object
  --final-lr FINAL_LR   The final learning rate to use for the sensitivity
                        analysis
  --steps-per-measurement STEPS_PER_MEASUREMENT
                        The number of steps (batches) to run for each
                        measurement
  --batch-size BATCH_SIZE
                        The batch size to use while training
#########
EXAMPLES
#########

##########
Example command for running approximated KS sensitivity analysis on mobilenet:
python integrations/pytorch/analysis.py pr_sensitivity \
    --approximate --arch-key mobilenet --dataset imagenet \
    --dataset-path ~/datasets/ILSVRC2012

##########
Example command for running one shot KS sens analysis on ssd300_resnet50 for coco:
python integrations/pytorch/analysis.py pr_sensitivity \
    --arch-key ssd300_resnet50 --dataset coco \
    --dataset-path ~/datasets/coco-detection

Note: Might need to install pycocotools using pip
##########
Example command for running LR sensitivity analysis on mobilenet:
python integrations/pytorch/analysis.py lr_sensitivity \
    --arch-key mobilenet --dataset imagenet \
    --dataset-path ~/datasets/ILSVRC2012 --batch-size 2
"""

import argparse
import json
import os
from typing import Any, List

from torch.nn import Module
from torch.optim import SGD
from torch.utils.data import DataLoader

import utils
from sparseml import get_main_logger
from sparseml.pytorch.models import ModelRegistry
from sparseml.pytorch.optim import (
    default_exponential_check_lrs,
    lr_loss_sensitivity,
    pruning_loss_sens_magnitude,
    pruning_loss_sens_one_shot,
)
from sparseml.pytorch.utils import PythonLogger, model_to_device


CURRENT_TASK = utils.Tasks.ANALYSIS
LOGGER = get_main_logger()


def pruning_loss_sensitivity(
    args: argparse.Namespace,
    model: Module,
    train_loader: DataLoader,
    save_dir: str,
    loggers: List[Any],
) -> None:
    """
    Utility function for pruning sensitivity analysis

    :params args : A Namespace object containing at-least the following keys
        approximate
    :param model: loaded model architecture to analyse
    :param train_loader: A DataLoader for training data
    :param save_dir: Directory to save results
    :param loggers: List of loggers to use during analysis
    """
    # loss setup
    if not args.approximate:
        loss = utils.get_loss_wrapper(args)
        LOGGER.info("created loss: {}".format(loss))
    else:
        loss = None

    # device setup
    if not args.approximate:
        module, device, device_ids = model_to_device(model, args.device)
    else:
        device = None

    # kernel sparsity analysis
    if args.approximate:
        analysis = pruning_loss_sens_magnitude(model)
    else:
        analysis = pruning_loss_sens_one_shot(
            model,
            train_loader,
            loss,
            device,
            args.steps_per_measurement,
            tester_loggers=loggers,
        )

    # saving and printing results
    LOGGER.info("completed...")
    LOGGER.info("Saving results in {}".format(save_dir))
    analysis.save_json(
        os.path.join(
            save_dir,
            "ks_approx_sensitivity.json"
            if args.approximate
            else "ks_one_shot_sensitivity.json",
        )
    )
    analysis.plot(
        os.path.join(
            save_dir,
            os.path.join(
                save_dir,
                "ks_approx_sensitivity.png"
                if args.approximate
                else "ks_one_shot_sensitivity.png",
            ),
        ),
        plot_integral=True,
    )
    analysis.print_res()


def lr_sensitivity(
    args: argparse.Namespace,
    model: Module,
    train_loader: DataLoader,
    save_dir: str,
) -> None:
    """
    Utility function to run learning rate sensitivity analysis

    :param args: A Namespace object containing at-least the following keys are
        init_lr, optim_args, arch_key, device, steps_per_measurement, final_lr
    :param model: loaded model architecture to analyse
    :param train_loader: A DataLoader for training data
    :param save_dir: Directory to save results
    """
    # optimizer setup
    optim = SGD(model.parameters(), lr=args.init_lr, **args.optim_args)
    LOGGER.info("created optimizer: {}".format(optim))

    # loss setup
    loss = utils.get_loss_wrapper(args)
    LOGGER.info("created loss: {}".format(loss))

    # device setup
    module, device, device_ids = model_to_device(model, args.device)

    # learning rate analysis
    LOGGER.info("running analysis: {}".format(loss))
    analysis = lr_loss_sensitivity(
        model,
        train_loader,
        loss,
        optim,
        device,
        args.steps_per_measurement,
        check_lrs=default_exponential_check_lrs(args.init_lr, args.final_lr),
        trainer_loggers=[PythonLogger()],
    )

    # saving and printing results
    LOGGER.info("completed...")
    LOGGER.info("Saving results in {}".format(save_dir))
    analysis.save_json(os.path.join(save_dir, "lr_sensitivity.json"))
    analysis.plot(os.path.join(save_dir, "lr_sensitivity.png"))
    analysis.print_res()


def parse_args() -> argparse.Namespace:
    """
    Utility method to parse command line arguments for analysis specific tasks
    """

    parser = argparse.ArgumentParser(
        description="Utility script to Run a kernel sparsity (pruning) or "
        "learning rate sensitivity analysis "
        "for a desired image classification architecture"
    )
    # DDP argument, necessary for launching via torch.distributed
    utils.add_local_rank(parser)

    subparsers = parser.add_subparsers(dest="command")
    pruning_sensitivity_parser = subparsers.add_parser(
        utils.Tasks.PR_SENSITIVITY.value,
        description="Run a kernel sparsity (pruning) analysis for a given model",
    )
    lr_sensitivity_parser = subparsers.add_parser(
        utils.Tasks.LR_SENSITIVITY.value,
        description="Run a learning rate sensitivity analysis for a desired image "
        "classification or detection architecture",
    )

    subtasks = [
        (pruning_sensitivity_parser, utils.Tasks.PR_SENSITIVITY),
        (lr_sensitivity_parser, utils.Tasks.LR_SENSITIVITY),
    ]

    for _parser, subtask in subtasks:
        utils.add_universal_args(parser=_parser)
        utils.add_device_args(parser=_parser)
        utils.add_workers_args(parser=_parser)
        utils.add_pin_memory_args(parser=_parser)

        _parser.add_argument(
            "--checkpoint-path",
            type=str,
            default=None,
            help=(
                "A path to a previous checkpoint to load the state from and "
                "resume the state for. If provided, pretrained will be ignored"
                ". If using a SparseZoo recipe, can also provide 'zoo' to load "
                "the base weights associated with that recipe"
            ),
        )

        if _parser == lr_sensitivity_parser:
            _parser.add_argument(
                "--init-lr",
                type=float,
                default=1e-5,
                help=("The initial learning rate to use for the sensitivity analysis"),
            )
            _parser.add_argument(
                "--optim-args",
                type=json.loads,
                default={},
                help="Additional args to be passed to the optimizer passed in"
                " as a json object",
            )
            _parser.add_argument(
                "--final-lr",
                type=float,
                default=0.5,
                help="The final learning rate to use for the sensitivity analysis",
            )

        _parser.add_argument(
            "--steps-per-measurement",
            type=int,
            default=15 if subtask == utils.Tasks.PR_SENSITIVITY else 20,
            help="The number of steps (batches) to run for each measurement",
        )
        _parser.add_argument(
            "--batch-size",
            type=int,
            required=subtask == utils.Tasks.LR_SENSITIVITY,
            default=64 if subtask == utils.Tasks.PR_SENSITIVITY else None,
            help="The batch size to use while training",
        )

        if _parser == pruning_sensitivity_parser:
            _parser.add_argument(
                "--approximate",
                action="store_true",
                help="True to approximate without running data through the model, "
                "otherwise will run a one shot analysis",
            )

    args = parser.parse_args()

    # set ddp args to default values
    args.local_rank = -1
    args.rank = -1
    args.world_size = 1
    args.is_main_process = True

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

    num_classes = utils.infer_num_classes(args_, train_dataset, val_dataset)

    model = utils.create_model(args_, num_classes)

    if args_.command == utils.Tasks.LR_SENSITIVITY.value:
        lr_sensitivity(args_, model, train_loader, save_dir)
    else:
        pruning_loss_sensitivity(args_, model, train_loader, save_dir, loggers)


if __name__ == "__main__":
    main()
