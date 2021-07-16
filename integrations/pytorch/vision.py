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
Perform optimization tasks on image classification and object detection models in
PyTorch including:
* Model pruning
* Quantization aware training
* Sparse transfer learning
* pruning sensitivity analysis
* learning rate sensitivity analysis
* ONNX export


##########
Command help:
usage: vision.py [-h] {train,export,pruning_sensitivity,lr_sensitivity} ...

Run tasks on classification and detection models and datasets using the
sparseml API

positional arguments:
  {train,export,pruning_sensitivity,lr_sensitivity}

optional arguments:
  -h, --help            show this help message and exit


##########
train command help:
usage: vision.py train [-h] --arch-key ARCH_KEY [--pretrained PRETRAINED]
                       [--pretrained-dataset PRETRAINED_DATASET]
                       [--checkpoint-path CHECKPOINT_PATH]
                       [--model-kwargs MODEL_KWARGS] --dataset DATASET
                       --dataset-path DATASET_PATH
                       [--dataset-kwargs DATASET_KWARGS]
                       [--model-tag MODEL_TAG] [--save-dir SAVE_DIR]
                       [--device DEVICE]
                       [--loader-num-workers LOADER_NUM_WORKERS]
                       [--loader-pin-memory LOADER_PIN_MEMORY]
                       [--init-lr INIT_LR] [--optim-args OPTIM_ARGS]
                       [--recipe-path RECIPE_PATH] [--sparse-transfer-learn]
                       [--eval-mode] --train-batch-size TRAIN_BATCH_SIZE
                       --test-batch-size TEST_BATCH_SIZE [--optim OPTIM]
                       [--logs-dir LOGS_DIR]
                       [--save-best-after SAVE_BEST_AFTER]
                       [--save-epochs SAVE_EPOCHS [SAVE_EPOCHS ...]]
                       [--use-mixed-precision] [--debug-steps DEBUG_STEPS]

Train and/or prune an image classification or detection architecture on a
dataset

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
  --checkpoint-path CHECKPOINT_PATH
                        A path to a previous checkpoint to load the state from
                        and resume the state for. If provided, pretrained will
                        be ignored. If using a SparseZoo recipe, can also
                        provide 'zoo' to load the base weights associated with
                        that recipe
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
                        Keyword arguments to be passed to dataset
                        constructor, should be given as a json object
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
  --init-lr INIT_LR     The initial learning rate to use while training, the
                        actual initial value used should be set by the
                        sparseml recipe
  --optim-args OPTIM_ARGS
                        Additional args to be passed to the optimizer passed
                        in as a json object
  --recipe-path RECIPE_PATH
                        The path to the yaml file containing the modifiers and
                        schedule to apply them with.  Can also provide a
                        SparseZoo stub prefixed with 'zoo:' with an optional
                        '?recipe_type=' argument"
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


##########
export command help:
usage: vision.py export [-h] --arch-key ARCH_KEY [--pretrained PRETRAINED]
                        [--pretrained-dataset PRETRAINED_DATASET]
                        [--checkpoint-path CHECKPOINT_PATH]
                        [--model-kwargs MODEL_KWARGS] --dataset DATASET
                        --dataset-path DATASET_PATH
                        [--dataset-kwargs DATASET_KWARGS]
                        [--model-tag MODEL_TAG] [--save-dir SAVE_DIR]
                        [--num-samples NUM_SAMPLES] [--onnx-opset ONNX_OPSET]
                        [--use-zipfile-serialization-if-available
                            USE_ZIPFILE_SERIALIZATION_IF_AVAILABLE]

Export a model to onnx as well as store sample inputs, outputs, and labels

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
  --checkpoint-path CHECKPOINT_PATH
                        A path to a previous checkpoint to load the state from
                        and resume the state for. If provided, pretrained will
                        be ignored
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
                        Keyword arguments to be passed to dataset
                        constructor, should be given as a json object
  --model-tag MODEL_TAG
                        A tag to use for the model for saving results under
                        save-dir, defaults to the model arch and dataset used
  --save-dir SAVE_DIR   The path to the directory for saving results
  --num-samples NUM_SAMPLES
                        The number of samples to export along with the model
                        onnx and pth files (sample inputs and labels as well
                        as the outputs from model execution)
  --onnx-opset ONNX_OPSET
                        The onnx opset to use for export. Default is 11
  --use-zipfile-serialization-if-available USE_ZIPFILE_SERIALIZATION_IF_AVAILABLE
                        for torch >= 1.6.0 only exports the Module's state
                        dict using the new zipfile serialization. Default is
                        True, has no affect on lower torch versions


##########
pruning_sensitivity command help:
usage: vision.py pruning_sensitivity [-h] --arch-key ARCH_KEY
                                     [--pretrained PRETRAINED]
                                     [--pretrained-dataset PRETRAINED_DATASET]
                                     [--checkpoint-path CHECKPOINT_PATH]
                                     [--model-kwargs MODEL_KWARGS] --dataset
                                     DATASET --dataset-path DATASET_PATH
                                     [--dataset-kwargs DATASET_KWARGS]
                                     [--model-tag MODEL_TAG]
                                     [--save-dir SAVE_DIR] [--device DEVICE]
                                     [--loader-num-workers LOADER_NUM_WORKERS]
                                     [--loader-pin-memory LOADER_PIN_MEMORY]
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
  --checkpoint-path CHECKPOINT_PATH
                        A path to a previous checkpoint to load the state from
                        and resume the state for. If provided, pretrained will
                        be ignored
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
                        Keyword arguments to be passed to dataset
                        constructor, should be given as a json object
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
  --steps-per-measurement STEPS_PER_MEASUREMENT
                        The number of steps (batches) to run for each
                        measurement
  --batch-size BATCH_SIZE
                        The batch size to use while training
  --approximate         True to approximate without running data through the
                        model, otherwise will run a one shot analysis


##########
lr_sensitivity command help:
usage: vision.py lr_sensitivity [-h] --arch-key ARCH_KEY
                                [--pretrained PRETRAINED]
                                [--pretrained-dataset PRETRAINED_DATASET]
                                [--checkpoint-path CHECKPOINT_PATH]
                                [--model-kwargs MODEL_KWARGS] --dataset
                                DATASET --dataset-path DATASET_PATH
                                [--dataset-kwargs DATASET_KWARGS]
                                [--model-tag MODEL_TAG] [--save-dir SAVE_DIR]
                                [--device DEVICE]
                                [--loader-num-workers LOADER_NUM_WORKERS]
                                [--loader-pin-memory LOADER_PIN_MEMORY]
                                [--init-lr INIT_LR] [--optim-args OPTIM_ARGS]
                                [--steps-per-measurement STEPS_PER_MEASUREMENT]
                                --batch-size BATCH_SIZE [--final-lr FINAL_LR]

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
  --checkpoint-path CHECKPOINT_PATH
                        A path to a previous checkpoint to load the state from
                        and resume the state for. If provided, pretrained will
                        be ignored
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
                        Keyword arguments to be passed to dataset
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
  --init-lr INIT_LR     The initial learning rate to use for the sensitivity
                        analysis
  --optim-args OPTIM_ARGS
                        Additional args to be passed to the optimizer passed
                        in as a json object
  --steps-per-measurement STEPS_PER_MEASUREMENT
                        The number of steps (batches) to run for each
                        measurement
  --batch-size BATCH_SIZE
                        The batch size to use while training
  --final-lr FINAL_LR   The final learning rate to use for the sensitivity
                        analysis
#########
EXAMPLES
#########

##########
Example command for pruning resnet50 on imagenet dataset:
python integrations/pytorch/vision.py train \
    --recipe-path ~/sparseml_recipes/pruning_resnet50.yaml \
    --arch-key resnet50 --dataset imagenet --dataset-path ~/datasets/ILSVRC2012 \
    --train-batch-size 256 --test-batch-size 1024

##########
Example command for transfer learning sparse mobilenet_v1 on an image folder dataset:
python integrations/pytorch/vision.py train \
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
integrations/pytorch/vision.py train \
--use-mixed-precision \
<VISION.PY TRAIN ARGUMENTS>

##########
Example command for exporting ResNet50:
python integrations/pytorch/vision.py export \
    --arch-key resnet50 --dataset imagenet --dataset-path ~/datasets/ILSVRC2012

##########
Example command for running approximated KS sensitivity analysis on mobilenet:
python integrations/pytorch/vision.py pruning_sensitivity \
    --approximate --arch-key mobilenet --dataset imagenet \
    --dataset-path ~/datasets/ILSVRC2012

##########
Example command for running one shot KS sens analysis on ssd300_resnet50 for coco:
python integrations/pytorch/vision.py pruning_sensitivity \
    --arch-key ssd300_resnet50 --dataset coco \
    --dataset-path ~/datasets/coco-detection

##########
Example command for running LR sensitivity analysis on mobilenet:
python integrations/pytorch/vision.py lr_sensitivity \
    --arch-key mobilenet --dataset imagenet \
    --dataset-path ~/datasets/ILSVRC2012 --batch-size 2
"""

import argparse
import json
import logging
import os
import time
from typing import Any, List, Tuple, Union

import torch
from torch.nn import Module
from torch.nn import functional as torch_functional
from torch.optim import SGD, Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


try:
    from torch.optim import RMSprop
except Exception:
    RMSprop = None
    logging.warning("RMSprop not available as an optimizer")

from sparseml import get_main_logger
from sparseml.pytorch.datasets import DatasetRegistry, ssd_collate_fn, yolo_collate_fn
from sparseml.pytorch.models import ModelRegistry
from sparseml.pytorch.optim import (
    ConstantPruningModifier,
    ScheduledModifierManager,
    ScheduledOptimizer,
    default_exponential_check_lrs,
    lr_loss_sensitivity,
    pruning_loss_sens_magnitude,
    pruning_loss_sens_one_shot,
)
from sparseml.pytorch.utils import (
    DEFAULT_LOSS_KEY,
    CrossEntropyLossWrapper,
    InceptionCrossEntropyLossWrapper,
    LossWrapper,
    ModuleDeviceContext,
    ModuleExporter,
    ModuleRunResults,
    ModuleTester,
    ModuleTrainer,
    PythonLogger,
    SSDLossWrapper,
    TensorBoardLogger,
    TopKAccuracy,
    YoloLossWrapper,
    default_device,
    early_stop_data_loader,
    get_prunable_layers,
    model_to_device,
    set_deterministic_seeds,
    tensor_sparsity,
    torch_distributed_zero_first,
)
from sparseml.utils import convert_to_bool, create_dirs
from sparsezoo import Zoo


LOGGER = get_main_logger()
TRAIN_COMMAND = "train"
EXPORT_COMMAND = "export"
PRUNING_SENSITVITY_COMMAND = "pruning_sensitivity"
LR_SENSITIVITY_COMMAND = "lr_sensitivity"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run tasks on classification and detection models and datasets "
        "using the sparseml API"
    )

    # DDP argument, necessary for launching via torch.distributed
    parser.add_argument(
        "--local_rank",  # DO NOT MODIFY
        type=int,
        default=-1,
        help=argparse.SUPPRESS,  # hide from help text
    )

    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser(
        TRAIN_COMMAND,
        description="Train and/or prune an image classification or detection "
        "architecture on a dataset",
    )
    export_parser = subparsers.add_parser(
        EXPORT_COMMAND,
        description="Export a model to onnx as well as "
        "store sample inputs, outputs, and labels",
    )
    pruning_sensitivity_parser = subparsers.add_parser(
        PRUNING_SENSITVITY_COMMAND,
        description="Run a kernel sparsity (pruning) analysis for a given model",
    )
    lr_sensitivity_parser = subparsers.add_parser(
        LR_SENSITIVITY_COMMAND,
        description="Run a learning rate sensitivity analysis for a desired image "
        "classification or detection architecture",
    )

    parsers = [
        train_parser,
        export_parser,
        pruning_sensitivity_parser,
        lr_sensitivity_parser,
    ]
    for par in parsers:
        # general arguments
        # model args
        par.add_argument(
            "--arch-key",
            type=str,
            required=True,
            help="The type of model to use, ex: resnet50, vgg16, mobilenet "
            "put as help to see the full list (will raise an exception with the list)",
        )
        par.add_argument(
            "--pretrained",
            type=str,
            default=True,
            help="The type of pretrained weights to use, "
            "default is true to load the default pretrained weights for the model. "
            "Otherwise should be set to the desired weights type: "
            "[base, optim, optim-perf]. "
            "To not load any weights set to one of [none, false]",
        )
        par.add_argument(
            "--pretrained-dataset",
            type=str,
            default=None,
            help="The dataset to load pretrained weights for if pretrained is set. "
            "Default is None which will load the default dataset for the architecture."
            " Ex can be set to imagenet, cifar10, etc",
        )
        checkpoint_path_help = (
            "A path to a previous checkpoint to load the state from and "
            "resume the state for. If provided, pretrained will be ignored"
        )
        if par == train_parser:
            checkpoint_path_help += (
                ". If using a SparseZoo recipe, can also provide 'zoo' to load "
                "the base weights associated with that recipe"
            )
        par.add_argument(
            "--checkpoint-path",
            type=str,
            default=None,
            help=checkpoint_path_help,
        )
        par.add_argument(
            "--model-kwargs",
            type=json.loads,
            default={},
            help="Keyword arguments to be passed to model constructor, should be "
            " given as a json object",
        )

        # dataset args
        par.add_argument(
            "--dataset",
            type=str,
            required=True,
            help="The dataset to use for training, "
            "ex: imagenet, imagenette, cifar10, etc. "
            "Set to imagefolder for a generic dataset setup "
            "with an image folder structure setup like imagenet or loadable by a "
            "dataset in sparseml.pytorch.datasets",
        )
        par.add_argument(
            "--dataset-path",
            type=str,
            required=True,
            help="The root path to where the dataset is stored",
        )
        par.add_argument(
            "--dataset-kwargs",
            type=json.loads,
            default={},
            help="Keyword arguments to be passed to dataset constructor, should be "
            " given as a json object",
        )

        # logging and saving
        par.add_argument(
            "--model-tag",
            type=str,
            default=None,
            help="A tag to use for the model for saving results under save-dir, "
            "defaults to the model arch and dataset used",
        )
        par.add_argument(
            "--save-dir",
            type=str,
            default="pytorch_vision",
            help="The path to the directory for saving results",
        )

        # task specific arguments
        if par in [train_parser, pruning_sensitivity_parser, lr_sensitivity_parser]:
            par.add_argument(
                "--device",
                type=str,
                default=default_device(),
                help=(
                    "The device to run on (can also include ids for data parallel), ex:"
                    " cpu, cuda, cuda:0,1"
                ),
            )
            par.add_argument(
                "--loader-num-workers",
                type=int,
                default=4,
                help="The number of workers to use for data loading",
            )
            par.add_argument(
                "--loader-pin-memory",
                type=bool,
                default=True,
                help="Use pinned memory for data loading",
            )

        if par in [train_parser, lr_sensitivity_parser]:
            par.add_argument(
                "--init-lr",
                type=float,
                default=1e-5 if par == lr_sensitivity_parser else 1e-9,
                help=(
                    "The initial learning rate to use for the sensitivity analysis"
                    if par == lr_sensitivity_parser
                    else "The initial learning rate to use while training, "
                    "the actual initial value used should be set by the sparseml recipe"
                ),
            )
            par.add_argument(
                "--optim-args",
                type=json.loads,
                default=(
                    {"momentum": 0.9, "nesterov": True, "weight_decay": 0.0001}
                    if par == train_parser
                    else {}
                ),
                help="Additional args to be passed to the optimizer passed in"
                " as a json object",
            )

        if par in [pruning_sensitivity_parser, lr_sensitivity_parser]:
            par.add_argument(
                "--steps-per-measurement",
                type=int,
                default=15 if par == pruning_sensitivity_parser else 20,
                help="The number of steps (batches) to run for each measurement",
            )
            par.add_argument(
                "--batch-size",
                type=int,
                required=par == lr_sensitivity_parser,
                default=64 if par == pruning_sensitivity_parser else None,
                help="The batch size to use while training",
            )

        if par == train_parser:
            par.add_argument(
                "--recipe-path",
                type=str,
                default=None,
                help="The path to the yaml file containing the modifiers and "
                "schedule to apply them with. Can also provide a SparseZoo stub "
                "prefixed with 'zoo:' with an optional '?recipe_type=' argument",
            )
            par.add_argument(
                "--sparse-transfer-learn",
                action="store_true",
                help=(
                    "Enable sparse transfer learning modifiers to enforce the sparsity "
                    "for already sparse layers. The modifiers are added to the "
                    "ones to be loaded from the recipe-path"
                ),
            )
            par.add_argument(
                "--eval-mode",
                action="store_true",
                help="Puts into evaluation mode so that the model can be "
                "evaluated on the desired dataset",
            )
            par.add_argument(
                "--train-batch-size",
                type=int,
                required=True,
                help="The batch size to use while training",
            )
            par.add_argument(
                "--test-batch-size",
                type=int,
                required=True,
                help="The batch size to use while testing",
            )
            par.add_argument(
                "--optim",
                type=str,
                default="SGD",
                help="The optimizer type to use, one of [SGD, Adam]",
            )
            par.add_argument(
                "--logs-dir",
                type=str,
                default=os.path.join("pytorch_vision_train", "tensorboard-logs"),
                help="The path to the directory for saving logs",
            )
            par.add_argument(
                "--save-best-after",
                type=int,
                default=-1,
                help="start saving the best validation result after the given "
                "epoch completes until the end of training",
            )
            par.add_argument(
                "--save-epochs",
                type=int,
                default=[],
                nargs="+",
                help="epochs to save checkpoints at",
            )
            par.add_argument(
                "--use-mixed-precision",
                action="store_true",
                help=(
                    "Trains model using mixed precision. Supported environments are "
                    "single GPU and multiple GPUs using DistributedDataParallel with "
                    "one GPU per process"
                ),
            )
            par.add_argument(
                "--debug-steps",
                type=int,
                default=-1,
                help="Amount of steps to run for training and testing for a debug mode",
            )

        if par == export_parser:
            par.add_argument(
                "--num-samples",
                type=int,
                default=100,
                help="The number of samples to export along with the model onnx "
                "and pth files (sample inputs and labels as well as the outputs "
                "from model execution)",
            )
            par.add_argument(
                "--onnx-opset",
                type=int,
                default=11,
                help="The onnx opset to use for export. Default is 11",
            )
            par.add_argument(
                "--use-zipfile-serialization-if-available",
                type=convert_to_bool,
                default=True,
                help="for torch >= 1.6.0 only exports the Module's state dict "
                "using the new zipfile serialization. Default is True, has no "
                "affect on lower torch versions",
            )

        if par == pruning_sensitivity_parser:
            par.add_argument(
                "--approximate",
                action="store_true",
                help="True to approximate without running data through the model, "
                "otherwise will run a one shot analysis",
            )

        if par == lr_sensitivity_parser:
            par.add_argument(
                "--final-lr",
                type=float,
                default=0.5,
                help="The final learning rate to use for the sensitivity analysis",
            )

    args = parser.parse_args()

    # append to dataset_kwargs based on arch type if coco or voc is the dataset
    if "preprocessing_type" not in args.dataset_kwargs and (
        "coco" in args.dataset.lower() or "voc" in args.dataset.lower()
    ):
        if "ssd" in args.arch_key.lower():
            args.dataset_kwargs["preprocessing_type"] = "ssd"
        elif "yolo" in args.arch_key.lower():
            args.dataset_kwargs["preprocessing_type"] = "yolo"

    return args


def parse_ddp_args(args):
    if args.command != TRAIN_COMMAND:
        # set ddp args to default values
        args.local_rank = -1
        args.rank = -1
        args.world_size = 1
        args.is_main_process = True
        return args

    args.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.rank = int(os.environ["RANK"]) if "RANK" in os.environ else -1
    args.is_main_process = args.rank in [-1, 0]  # non DDP execution or 0th DDP process

    # modify training batch size for give world size
    assert args.train_batch_size % args.world_size == 0, (
        "Invalid training batch size for world size {}"
        " given batch size {}. world size must divide training batch size evenly."
    ).format(args.world_size, args.train_batch_size)
    args.train_batch_size = args.train_batch_size // args.world_size

    return args


def _get_save_dir_and_loggers(args) -> Tuple[Union[str, None], List]:
    if args.is_main_process:
        save_dir = os.path.abspath(os.path.expanduser(args.save_dir))
        logs_dir = (
            os.path.abspath(os.path.expanduser(os.path.join(args.logs_dir)))
            if args.command == TRAIN_COMMAND
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
        if args.command == TRAIN_COMMAND:
            logs_dir = os.path.join(logs_dir, model_id)
            create_dirs(logs_dir)
            loggers.append(TensorBoardLogger(log_path=logs_dir))
        LOGGER.info("Model id is set to {}".format(model_id))
    else:
        # do not log for non main processes
        save_dir = None
        loggers = []
    return save_dir, loggers


def _get_collate_fn(args):
    if (
        "ssd" not in args.arch_key.lower() and "yolo" not in args.arch_key.lower()
    ) or args.command == EXPORT_COMMAND:
        # default collate function
        return None
    return ssd_collate_fn if "ssd" in args.arch_key.lower() else yolo_collate_fn


def _create_train_dataset_and_loader(
    args, image_size: Tuple[int, ...]
) -> Tuple[Any, Any]:
    # create train dataset if it should be ran for this flow, otherwise return None
    if (
        args.command == EXPORT_COMMAND
        or args.command == PRUNING_SENSITVITY_COMMAND
        and args.approximate
        or args.command == TRAIN_COMMAND
        and args.eval_mode
    ):
        return None, None

    with torch_distributed_zero_first(args.local_rank):  # only download once locally
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
    batch_size = (
        args.train_batch_size if args.command == TRAIN_COMMAND else args.batch_size
    )

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
    args, image_size: Tuple[int, ...]
) -> Tuple[Any, Any]:
    if (
        args.command == PRUNING_SENSITVITY_COMMAND
        or args.command == LR_SENSITIVITY_COMMAND
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
        is_training = args.command == TRAIN_COMMAND
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.test_batch_size if is_training else 1,
            shuffle=False,
            num_workers=args.loader_num_workers if is_training else 1,
            pin_memory=args.loader_pin_memory if is_training else False,
            collate_fn=_get_collate_fn(args),
        )
        if args.command == EXPORT_COMMAND:
            val_loader = early_stop_data_loader(
                val_loader, args.num_samples if args.num_samples > 1 else 1
            )
        LOGGER.info("created val_dataset: {}".format(val_dataset))
    else:
        val_loader = None  # only val dataset needed to get the number of classes
    return val_dataset, val_loader


def _get_loss_wrapper(args, training=False):
    if "ssd" in args.arch_key.lower():
        return SSDLossWrapper()
    if "yolo" in args.arch_key.lower():
        return YoloLossWrapper()

    extras = {"top1acc": TopKAccuracy(1), "top5acc": TopKAccuracy(5)}
    if args.command == TRAIN_COMMAND:
        return (
            CrossEntropyLossWrapper(extras)
            if training and "inception" not in args.arch_key
            else InceptionCrossEntropyLossWrapper(extras)
        )
    return LossWrapper(loss_fn=torch_functional.cross_entropy, extras=extras)


def _create_scheduled_optimizer(
    args,
    model: Module,
    train_loader: DataLoader,
    loggers: List[Any],
) -> Tuple[int, ScheduledOptimizer, ScheduledModifierManager]:
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


def _save_model_training(
    model: Module,
    optim: Optimizer,
    input_shape: Tuple[int, ...],
    save_name: str,
    save_dir: str,
    epoch: int,
    val_res: Union[ModuleRunResults, None],
    convert_qat: bool = False,
):
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


def _save_recipe(
    recipe_manager: ScheduledModifierManager,
    save_dir: str,
):

    recipe_save_path = os.path.join(save_dir, "recipe.yaml")
    recipe_manager.save(recipe_save_path)
    LOGGER.info(f"Saved recipe to {recipe_save_path}")


def train(args, model, train_loader, val_loader, input_shape, save_dir, loggers):
    # loss setup
    val_loss = _get_loss_wrapper(args, training=True)
    LOGGER.info("created loss for validation: {}".format(val_loss))
    train_loss = _get_loss_wrapper(args, training=True)
    LOGGER.info("created loss for training: {}".format(train_loss))

    # training setup
    if not args.eval_mode:
        epoch, optim, manager = _create_scheduled_optimizer(
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
        _save_recipe(recipe_manager=manager, save_dir=save_dir)
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
                    _save_model_training(
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
                _save_model_training(
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
            _save_model_training(
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


def export(args, model, val_loader, save_dir):
    exporter = ModuleExporter(model, save_dir)

    # export PyTorch state dict
    LOGGER.info("exporting pytorch in {}".format(save_dir))
    exporter.export_pytorch(
        use_zipfile_serialization_if_available=(
            args.use_zipfile_serialization_if_available
        )
    )
    onnx_exported = False

    for batch, data in tqdm(
        enumerate(val_loader),
        desc="Exporting samples",
        total=args.num_samples if args.num_samples > 1 else 1,
    ):
        if not onnx_exported:
            # export onnx file using first sample for graph freezing
            LOGGER.info("exporting onnx in {}".format(save_dir))
            exporter.export_onnx(data[0], opset=args.onnx_opset, convert_qat=True)
            onnx_exported = True

        if args.num_samples > 0:
            exporter.export_samples(
                sample_batches=[data[0]], sample_labels=[data[1]], exp_counter=batch
            )


def pruning_loss_sensitivity(args, model, train_loader, save_dir, loggers):
    # loss setup
    if not args.approximate:
        loss = _get_loss_wrapper(args)
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


def lr_sensitivity(args, model, train_loader, save_dir, loggers):
    # optimizer setup
    optim = SGD(model.parameters(), lr=args.init_lr, **args.optim_args)
    LOGGER.info("created optimizer: {}".format(optim))

    # loss setup
    loss = _get_loss_wrapper(args)
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


def main(args):
    """
    load model
    load dataset
    split by command
    """

    # SET UP LOGGING, DATASETS, AND MODEL
    # logging and saving setup
    save_dir, loggers = _get_save_dir_and_loggers(args)

    # dataset creation
    input_shape = ModelRegistry.input_shape(args.arch_key)
    image_size = input_shape[1]  # assume shape [C, S, S] where S is the image size

    train_dataset, train_loader = _create_train_dataset_and_loader(args, image_size)
    val_dataset, val_loader = _create_val_dataset_and_loader(args, image_size)

    # model creation
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

    # RUN COMMAND SPECIFIC TASKS
    if args.command == TRAIN_COMMAND:
        train(args, model, train_loader, val_loader, input_shape, save_dir, loggers)
    if args.command == EXPORT_COMMAND:
        export(args, model, val_loader, save_dir)
    if args.command == PRUNING_SENSITVITY_COMMAND:
        pruning_loss_sensitivity(args, model, train_loader, save_dir, loggers)
    if args.command == LR_SENSITIVITY_COMMAND:
        lr_sensitivity(args, model, train_loader, save_dir, loggers)

    # add sleep to make sure all background processes have finished,
    # ex tensorboard writing
    time.sleep(5)


if __name__ == "__main__":
    args_ = parse_args()
    args_ = parse_ddp_args(args_)

    # initialize DDP process, set deterministic seeds
    if args_.local_rank != -1:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        set_deterministic_seeds(0)

    main(args_)
