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
Perform tasks on image classification models in Keras including:
* Model training/pruning
* Model evaluation
* ONNX export


##########
train command help:
usage: classification.py train [-h] --arch-key ARCH_KEY
                               [--pretrained PRETRAINED]
                               [--pretrained-dataset PRETRAINED_DATASET]
                               [--checkpoint-path CHECKPOINT_PATH]
                               [--model-kwargs MODEL_KWARGS] --dataset DATASET
                               --dataset-path DATASET_PATH
                               [--dataset-kwargs DATASET_KWARGS]
                               [--model-tag MODEL_TAG] [--save-dir SAVE_DIR]
                               [--dataset-parallel-calls DATASET_PARALLEL_CALLS]
                               [--shuffle-buffer-size SHUFFLE_BUFFER_SIZE]
                               [--recipe-path RECIPE_PATH]
                               [--sparse-transfer-learn] [--eval-mode]
                               --train-batch-size TRAIN_BATCH_SIZE
                               --test-batch-size TEST_BATCH_SIZE
                               [--logs-dir LOGS_DIR]
                               [--save-best-after SAVE_BEST_AFTER]
                               [--save-epochs SAVE_EPOCHS [SAVE_EPOCHS ...]]
                               [--init-lr INIT_LR] [--optim-args OPTIM_ARGS]
Train and/or prune an image classification model
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
                        be ignored. Specify "zoo" to load a model from SparseZoo
                        based on other information such as arch key and pretrained
                        dataset.
  --model-kwargs MODEL_KWARGS
                        key word arguments to be passed to model constructor,
                        should be given as a json object
  --dataset DATASET     The dataset to use for training, ex: imagenet,
                        imagenette, cifar10, etc. Set to imagefolder for a
                        generic dataset setup with an image folder structure
                        setup like imagenet or loadable by a dataset in
                        sparseml.keras.datasets
  --dataset-path DATASET_PATH
                        The root path to where the dataset is stored
  --dataset-kwargs DATASET_KWARGS
                        Key word arguments to be passed to dataset
                        constructor, should be given as a json object
  --transfer-class-type TRANSFER_CLASS_TYPE
                        Type of the target class for transfer ("single" for softmax
                        function or "multi" for sigmoid function); takes effects only
                        if the specified dataset is different from the pretrained
                        dataset, implying a transfer learning usecase
  --transfer-top-layer TRANSFER_TOP_LAYER
                        Name of the layer which a custom dense layer will be put on;
                        None by default to imply the layer right under the top of the
                        original model will be used (in other words, layer[-2])
  --model-tag MODEL_TAG
                        A tag to use for the model for saving results under
                        save-dir, defaults to the model arch and dataset used
  --save-dir SAVE_DIR   The path to the directory for saving results
  --train-dataset-parallel-calls DATASET_PARALLEL_CALLS
                        the number of parallel workers for train dataset loading
  --train-shuffle-buffer-size SHUFFLE_BUFFER_SIZE
                        Shuffle buffer size for train dataset loading
  --train-prefetch-buffer-size PREFETCH_BUFFER_SIZE
                        Prefetch buffer size for train dataset loading
  --val-dataset-parallel-calls DATASET_PARALLEL_CALLS
                        the number of parallel workers for val dataset loading
  --val-shuffle-buffer-size SHUFFLE_BUFFER_SIZE
                        Shuffle buffer size for val dataset loading
  --val-prefetch-buffer-size PREFETCH_BUFFER_SIZE
                        Prefetch buffer size for val dataset loading
  --recipe-path RECIPE_PATH
                        The path to the yaml file containing the modifiers and
                        schedule to apply them with. If set to
                        'transfer_learning', then will create a schedule to
                        enable sparse transfer learning. Recipe from SparseZoo
                        could be specified using model stub starting with "zoo:"
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
  --logs-dir LOGS_DIR   The path to the directory for saving logs
  --save-best-after SAVE_BEST_AFTER
                        start saving the best validation result after the
                        given epoch completes until the end of training
  --save-epochs SAVE_EPOCHS [SAVE_EPOCHS ...]
                        epochs to save checkpoints at
  --init-lr INIT_LR     The initial learning rate to use while training, the
                        actual initial value used should be set by the
                        sparseml recipe
  --optim-args OPTIM_ARGS
                        Additional args to be passed to the optimizer passed
                        in as a json object
################
Evaluate command help:
usage: classification.py evaluate [-h] --arch-key ARCH_KEY
                               [--pretrained PRETRAINED]
                               [--pretrained-dataset PRETRAINED_DATASET]
                               [--checkpoint-path CHECKPOINT_PATH]
                               [--model-kwargs MODEL_KWARGS] --dataset DATASET
                               --dataset-path DATASET_PATH
                               [--dataset-kwargs DATASET_KWARGS]
                               [--dataset-parallel-calls DATASET_PARALLEL_CALLS]
                               --test-batch-size TEST_BATCH_SIZE
Evaluate an image classification model
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
                        be ignored. Specify "zoo" to load a model from SparseZoo
                        based on other information such as arch key and pretrained
                        dataset.
  --dataset DATASET     The dataset to use for training, ex: imagenet,
                        imagenette, cifar10, etc. Set to imagefolder for a
                        generic dataset setup with an image folder structure
                        setup like imagenet or loadable by a dataset in
                        sparseml.keras.datasets
  --dataset-path DATASET_PATH
                        The root path to where the dataset is stored
  --dataset-kwargs DATASET_KWARGS
                        kew word arguments to be passed to dataset
                        constructor, should be given as a json object
  --test-batch-size TEST_BATCH_SIZE
                        The batch size to use while evaluating
################
Export command help:
usage: classification.py export [-h] --arch-key ARCH_KEY
                                [--pretrained PRETRAINED]
                                [--pretrained-dataset PRETRAINED_DATASET]
                                [--checkpoint-path CHECKPOINT_PATH]
                                --dataset-path DATASET_PATH
                                [--dataset-kwargs DATASET_KWARGS]
                                [--dataset-parallel-calls DATASET_PARALLEL_CALLS]
                                [--num-samples NUM_SAMPLES]
                                [--onnx-opset OPSET]
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
                        be ignored. Specify "zoo" to load a model from SparseZoo
                        based on other information such as arch key and pretrained
                        dataset.
  --dataset DATASET     The dataset to use for training, ex: imagenet,
                        imagenette, cifar10, etc. Set to imagefolder for a
                        generic dataset setup with an image folder structure
                        setup like imagenet or loadable by a dataset in
                        sparseml.keras.datasets
  --dataset-path DATASET_PATH
                        The root path to where the dataset is stored
  --dataset-kwargs DATASET_KWARGS
                        key word arguments to be passed to dataset
                        constructor, should be given as a json object
  --num-samples NUM_SAMPLES
                        Number of samples to generate, default to 100
  --onnx-opset OPSET
                        The ONNX opset to export to, default to 11
  --save-dir SAVE_DIR   The path to the directory for saving results
  --export-debug-mode EXPORT_DEBUG
                        The debug mode for ONNX export (passed into keras2onnx).
                        Default to False.


############
EXAMPLES
############

#############
Example command for fine-tuning a ResNet50 model from Keras Application, pretrained on
Imagenet, for the Imagenette dataset. The recipe could be used to define learning rate
schedule. Note that there're no pruning performed from this experiment.
python scripts/keras_classification.py train \
     --arch-key keras_applications.ResNet50 \
     --pretrained-dataset imagenet \
     --dataset imagenette \
     --dataset-path <DATASET_DIR> \
     --train-batch-size 64 \
     --recipe-path <RECIPE_PATH>
############
Example command for pruning a model with a recipe. Assuming that a ResNet50 model
from Keras Application was fine-tuned for the Imagenette dataset. The following
command specifies the model checkpoint after fine-tuning, and a recipe that could
be used next to prune its weights:
python scripts/keras_classification.py train \
     --arch-key keras_applications.ResNet50 \
     --checkpoint-path <MODEL_CHECKPOINT> \
     --dataset imagenette \
     --dataset-path /hdd/datasets/imagenette \
     --train-batch-size 64 \
     --recipe-path <RECIPE_PATH>

In the above command, specifying "zoo" for --checkpoint-path forces loading the
base model from SparseZoo. The following command prunes a ResNet50 from Keras
Applications fine-tuned for Imagenette dataset, using a recipe from SparseZoo:

python scripts/keras_classification.py train \
     --arch-key keras_applications.ResNet50 \
     --checkpoint-path zoo \
     --pretrained-dataset imagenette \
     --dataset imagenette \
     --dataset-path <DATASET_DIR> \
     --train-batch-size 64 \
     --recipe-path zoo:cv/classification/resnet_v1-50/keras/\
        keras.applications/imagenette/pruned-moderate?recipe_type=original
############
Example command for loading a ResNet50 model from a checkpoint, then
evaluate it on the validation of the Imagenette dataset.
python scripts/keras_classification.py evaluate \
     --arch-key keras_applications.ResNet50 \
     --checkpoint-path <MODEL_CHECKPOINT> \
     --dataset imagenette \
     --dataset-path /hdd/datasets/imagenette \
     --test-batch-size 64
############
Example command for loading a ResNet50 model from a checkpoint, then export it
into ONNX format. It also generates a number of inputs, outputs and labels.
python scripts/keras_classification.py export \
     --arch-key keras_applications.ResNet50 \
     --checkpoint-path <MODEL_CHECKPOINT> \
     --dataset imagenette \
     --dataset-path /hdd/datasets/imagenette \
     --num-samples <NUM_SAMPLES>
     --save-dir <EXPORT_DIR>
"""

import argparse
import inspect
import json
import math
import os
from typing import Optional, Tuple

import numpy
import tensorflow

from sparseml import get_main_logger
from sparseml.keras.datasets import Dataset, DatasetRegistry
from sparseml.keras.models import ModelRegistry
from sparseml.keras.optim import ScheduledModifierManager, remove_pruning_masks
from sparseml.keras.utils import (
    LossesAndMetricsLoggingCallback,
    ModelExporter,
    TensorBoardLogger,
    keras,
)
from sparseml.utils import create_dirs


LOGGER = get_main_logger()
TRAIN_COMMAND = "train"
EVAL_COMMAND = "evaluate"
EXPORT_COMMAND = "export"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run tasks on classification models and datasets "
        "using the sparseml API"
    )

    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser(
        TRAIN_COMMAND,
        description="Train and/or prune an image classification model",
    )
    eval_parser = subparsers.add_parser(
        EVAL_COMMAND,
        description="Evaluate an image classification model on a dataset",
    )
    export_parser = subparsers.add_parser(
        EXPORT_COMMAND,
        description="Export a model to onnx as well as "
        "store sample inputs, outputs, and labels",
    )

    parsers = [
        train_parser,
        eval_parser,
        export_parser,
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
        par.add_argument(
            "--checkpoint-path",
            type=str,
            default=None,
            help="A path to a previous checkpoint to load the state from and "
            "resume the state for. If provided, pretrained will be ignored",
        )
        par.add_argument(
            "--model-kwargs",
            type=json.loads,
            default={},
            help="kew word arguments to be passed to model constructor, should be "
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
            "dataset in sparseml.keras.datasets",
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
            help="kew word arguments to be passed to dataset constructor, should be "
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
            default="keras_classification",
            help="The path to the directory for saving results",
        )

    # task specific arguments: training
    train_parser.add_argument(
        "--dataset-parallel-calls",
        type=int,
        default=4,
        help="the number of parallel workers for dataset loading",
    )
    train_parser.add_argument(
        "--train-shuffle-buffer-size",
        type=int,
        default=None,
        help="Shuffle buffer size for dataset loading",
    )
    train_parser.add_argument(
        "--train-prefetch-buffer-size",
        type=int,
        default=None,
        help="Prefetch buffer size for train dataset loading",
    )
    train_parser.add_argument(
        "--test-prefetch-buffer-size",
        type=int,
        default=None,
        help="Prefetch buffer size for test dataset loading",
    )
    train_parser.add_argument(
        "--recipe-path",
        type=str,
        default=None,
        help="The path to the yaml file containing the modifiers and "
        "schedule to apply them with. If set to 'transfer_learning', "
        "then will create a schedule to enable sparse transfer learning",
    )
    train_parser.add_argument(
        "--transfer-class-type",
        type=str,
        default="single",
        help="Type of target function: 'single' for softmax, 'multi' for sigmoid "
        "or 'linear' for linear function. This option takes effects only when the "
        "specified dataset is different from the pretrained dataset, implying a "
        "transfer learning usecase.",
    )
    train_parser.add_argument(
        "--transfer-top-layer",
        type=str,
        default=None,
        help="Name of the layer which a custom dense layer will be put on. Default "
        "to None to imply the layer right under the top of the original model "
        "will be used (in other words, layer[-2])",
    )
    train_parser.add_argument(
        "--train-batch-size",
        type=int,
        required=True,
        help="The batch size to use while training",
    )
    train_parser.add_argument(
        "--test-batch-size",
        type=int,
        required=False,
        help="The batch size to use while testing; default to the train " "batch size",
    )
    train_parser.add_argument(
        "--log-dir",
        type=str,
        default=os.path.join("keras_classification_train", "tensorboard-logs"),
        help="The path to the directory for saving logs",
    )
    train_parser.add_argument(
        "--log-epoch",
        type=bool,
        default=True,
        help="Whether logging should be performed at the end of each epoch",
    )
    train_parser.add_argument(
        "--log-batch",
        type=bool,
        default=False,
        help="Whether logging should be performed at the end of each training " "batch",
    )
    train_parser.add_argument(
        "--log-steps",
        type=int,
        default=-1,
        help="Whether logging should be performed after every specified number "
        "of steps",
    )
    train_parser.add_argument(
        "--save-best-only",
        type=bool,
        default=True,
        help="Save model only with better monitored metric",
    )
    train_parser.add_argument(
        "--optim",
        type=str,
        default="SGD",
        help="The optimizer type to use, e.g., 'Adam', 'SGD' etc",
    )
    train_parser.add_argument(
        "--optim-args",
        type=json.loads,
        default={"momentum": 0.9, "nesterov": True},
        # default={},
        help="Additional args to be passed to the optimizer passed in"
        " as a json object",
    )
    train_parser.add_argument(
        "--run-eagerly",
        type=bool,
        default=True,
        help="Run training in eager execution mode",
    )

    # task specific arguments: evaluation
    eval_parser.add_argument(
        "--test-batch-size",
        type=int,
        required=False,
        help="The batch size to use while testing; default to the train " "batch size",
    )
    eval_parser.add_argument(
        "--dataset-parallel-calls",
        type=int,
        default=4,
        help="the number of parallel workers for dataset loading",
    )
    eval_parser.add_argument(
        "--test-prefetch-buffer-size",
        type=int,
        default=None,
        help="Prefetch buffer size for test dataset loading",
    )

    # task specific arguments: export
    export_parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="The number of samples to export along with the model onnx "
        "and pth files (sample inputs and labels as well as the outputs "
        "from model execution)",
    )
    export_parser.add_argument(
        "--onnx-opset",
        type=int,
        default=11,
        help="The onnx opset to use for export. Default is 11",
    )
    export_parser.add_argument(
        "--export-debug-mode",
        type=bool,
        default=False,
        help="The debug mode for ONNX export (passed into keras2onnx). "
        "Default to False.",
    )

    return parser.parse_args()


def setup_save_and_log_dirs(args) -> Tuple[str, Optional[str]]:
    # Saving dir setup
    save_dir = os.path.abspath(os.path.expanduser(args.save_dir))
    if not args.model_tag:
        model_tag = "{}@{}".format(args.arch_key, args.dataset)
        model_id = model_tag
        model_inc = 0

        while os.path.exists(os.path.join(save_dir, model_id)):
            model_inc += 1
            model_id = "{}__{:02d}".format(model_tag, model_inc)
    else:
        model_id = args.model_tag

    save_dir = os.path.join(save_dir, model_id)
    create_dirs(save_dir)
    LOGGER.info("Model directory is set to {}".format(save_dir))

    # log dir setup
    log_dir = (
        os.path.abspath(os.path.expanduser(args.log_dir))
        if args.command == TRAIN_COMMAND
        else None
    )
    if args.command == TRAIN_COMMAND:
        log_dir = os.path.join(log_dir, model_id)
        create_dirs(log_dir)
        LOGGER.info("Logging directory is set to {}".format(log_dir))
    else:
        log_dir = None
    return save_dir, log_dir


def create_dataset(
    args, train: bool, image_size: Tuple[int, int]
) -> Tuple[Dataset, int]:
    kwargs = args.dataset_kwargs
    dataset = DatasetRegistry.create(
        args.dataset,
        root=args.dataset_path,
        train=train,
        image_size=image_size,
        **kwargs,
    )
    LOGGER.info(
        "created {} dataset: {}, images to resize to {}".format(
            "train" if train else "val", dataset, image_size
        )
    )

    # get num_classes
    if args.dataset == "imagefolder":
        num_classes = dataset.num_classes
    else:
        dataset_attributes = DatasetRegistry.attributes(args.dataset)
        num_classes = dataset_attributes["num_classes"]

    return dataset, num_classes


def build_dataset(
    args, dataset: Dataset, train: bool = True
) -> tensorflow.data.Dataset:
    test_batch_size = (
        args.test_batch_size if args.test_batch_size else args.train_batch_size
    )
    batch_size = args.train_batch_size if train else test_batch_size
    if train:
        shuffle_buffer_size = (
            args.train_shuffle_buffer_size
            if args.train_shuffle_buffer_size
            else dataset.num_images
        )
    else:
        shuffle_buffer_size = None

    if train:
        prefetch_buffer_size = (
            args.train_prefetch_buffer_size
            if args.train_prefetch_buffer_size
            else batch_size * 8
        )
    else:
        prefetch_buffer_size = (
            args.test_prefetch_buffer_size
            if args.test_prefetch_buffer_size
            else batch_size * 8
        )
    built_dataset = dataset.build(
        batch_size,
        repeat_count=1,
        shuffle_buffer_size=shuffle_buffer_size,
        prefetch_buffer_size=prefetch_buffer_size,
        num_parallel_calls=args.dataset_parallel_calls,
    )
    return built_dataset


def create_model(args, input_shape, num_classes):
    kwargs = args.model_kwargs

    if args.checkpoint_path == "zoo":
        zoo_model = ModelRegistry.create_zoo_model(
            args.arch_key, args.pretrained, args.pretrained_dataset
        )
        model_file_paths = zoo_model.download_framework_files(extensions=[".h5"])
        if not model_file_paths:
            model_file_paths = zoo_model.download_framework_files(extensions=[".tf"])
        if not model_file_paths:
            raise RuntimeError("Error downloading model from SparseZoo")
        model_file_path = model_file_paths[0]
        model = keras.models.load_model(model_file_path)
    else:
        model = ModelRegistry.create(
            args.arch_key,
            args.pretrained,
            args.checkpoint_path,
            args.pretrained_dataset,
            input_shape=input_shape,
            **kwargs,
        )

    if (
        args.pretrained_dataset is not None
        and args.dataset is not None
        and args.pretrained_dataset != args.dataset
    ):
        # Set up the model for fine-tuning
        # The trainablility of model's parameters should be controlled
        # through recipe; we therefore do not set "model.trainable" to False here.
        if args.transfer_class_type == "single":
            activation = "softmax"
        elif args.transfer_class_type == "multi":
            activation = "sigmoid"
        elif args.transfer_class_type == "linear":
            activation = None
        else:
            raise ValueError(
                "Unknown transfer_class_type; expected: 'single', 'softmax' or 'linear'"
            )
        if args.transfer_top_layer is not None:
            transfer_top_layer = None
            for i in list(range(len(model.layers)))[::-1]:
                if model.layers[i].name == args.transfer_top_layer:
                    transfer_top_layer = args.transfer_top_layer
                    break
            if transfer_top_layer is None:
                raise ValueError(
                    "Top layer for transfer learning {} not found".format(
                        args.transfer_top_layer
                    )
                )
        else:
            transfer_top_layer = model.layers[-2]
        new_output = keras.layers.Dense(num_classes, activation=activation)(
            transfer_top_layer.output
        )
        model = keras.Model(model.input, new_output)

    return model


def create_optimizer(args):
    optim_const = {}
    for opt_name in dir(keras.optimizers):
        opt = getattr(keras.optimizers, opt_name)
        if inspect.isclass(opt) and opt_name != "Optimizer":
            optim_const[opt_name] = opt
    if args.optim not in optim_const:
        raise ValueError(
            "Unknown optimizer {}. Supported: {}".format(
                args.optim, list(optim_const.keys())
            )
        )
    optimizer = optim_const[args.optim](**args.optim_args)
    return optimizer


def get_logging_update_freq(args):
    if args.log_steps > 0:
        update_freq = args.log_steps
    elif args.log_batch:
        update_freq = "batch"
    elif args.log_epoch:
        update_freq = "epoch"
    else:
        update_freq = None
    return update_freq


def handle_train_command(args):
    save_dir, log_dir = setup_save_and_log_dirs(args)

    input_shape = ModelRegistry.input_shape(args.arch_key)
    image_size = (input_shape[0], input_shape[1])

    # Load datasets
    train_dataset, num_classes = create_dataset(args, train=True, image_size=image_size)
    num_train_images = train_dataset.num_images
    train_dataset = build_dataset(args, train_dataset, train=True)

    val_dataset, _ = create_dataset(args, train=False, image_size=image_size)
    val_dataset = build_dataset(args, val_dataset, train=False)

    # Create model
    model = create_model(args, input_shape, num_classes=num_classes)

    # Create optimizer
    optimizer = create_optimizer(args)

    # Logging
    if log_dir:
        update_freq = get_logging_update_freq(args)
        if update_freq is None:
            raise ValueError(
                "Logging requires update frequency to take effect; use either "
                "'log-epoch', 'log-batch' or 'log-steps' option."
            )
        loggers = TensorBoardLogger(log_dir=log_dir, update_freq=update_freq)
    else:
        loggers = []

    # Model saving
    checkpoint_filepath = os.path.join(
        save_dir, "model.{epoch:02d}-{val_accuracy:.2f}.tf"
    )
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    # Manager
    manager = ScheduledModifierManager.from_yaml(args.recipe_path)

    # Enhance model
    steps_per_epoch = math.ceil(num_train_images / args.train_batch_size)
    model, optimizer, callbacks = manager.modify(
        model, optimizer, steps_per_epoch, loggers=loggers
    )
    if loggers:
        callbacks.append(LossesAndMetricsLoggingCallback(loggers))
    callbacks.append(checkpoint_callback)

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=optimizer,
        metrics=["accuracy"],
        run_eagerly=args.run_eagerly,
    )
    model.fit(
        train_dataset,
        epochs=manager.max_epochs,
        callbacks=callbacks,
        validation_data=val_dataset,
    )


def handle_eval_command(args):
    input_shape = ModelRegistry.input_shape(args.arch_key)
    image_size = (input_shape[0], input_shape[1])

    # Load dataset
    val_dataset, num_classes = create_dataset(args, train=False, image_size=image_size)
    val_dataset = build_dataset(args, val_dataset, train=False)

    # Create model
    model = create_model(args, input_shape, num_classes=num_classes)

    # Evaluate
    result = model.evaluate(val_dataset)
    print(dict(zip(model.metrics_names, result)))


def handle_export_command(args):
    input_shape = ModelRegistry.input_shape(args.arch_key)
    image_size = (input_shape[0], input_shape[1])

    train_dataset, num_classes = create_dataset(args, train=True, image_size=image_size)
    train_dataset = train_dataset.build(args.num_samples)

    model = create_model(args, input_shape, num_classes=num_classes)
    model = remove_pruning_masks(model)
    exporter = ModelExporter(model, args.save_dir)
    exporter.export_onnx(opset=args.onnx_opset, debug_mode=args.export_debug_mode)

    for img_batch, label_batch in train_dataset.take(1):
        output_batch = model(img_batch)
        numpy.save(os.path.join(args.save_dir, "inputs.npy"), img_batch)
        numpy.save(os.path.join(args.save_dir, "outputs.npy"), output_batch)
        numpy.save(os.path.join(args.save_dir, "labels.npy"), label_batch)


def main(args):
    if args.command == TRAIN_COMMAND:
        handle_train_command(args)
    elif args.command == EVAL_COMMAND:
        handle_eval_command(args)
    elif args.command == EXPORT_COMMAND:
        handle_export_command(args)
    else:
        raise ValueError("Unknown command passed in")


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
