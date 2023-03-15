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
Perform optimization tasks on image classification tensorflow_v1 including:
* Model training
* Model pruning
* Sparse transfer learning
* pruning sensitivity analysis
* ONNX export


##########
Command help:
usage: classification.py [-h] {train,export,pruning_sensitivity} ...

Run tasks on classification models and datasets using the sparseml API

positional arguments:
  {train,export,pruning_sensitivity}

optional arguments:
  -h, --help            show this help message and exit


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
                        be ignored
  --model-kwargs MODEL_KWARGS
                        kew word arguments to be passed to model constructor,
                        should be given as a json object
  --dataset DATASET     The dataset to use for training, ex: imagenet,
                        imagenette, cifar10, etc. Set to imagefolder for a
                        generic dataset setup with an image folder structure
                        setup like imagenet or loadable by a dataset in
                        sparseml.tensorflow_v1.datasets
--dataset-path DATASET_PATH
                        The root path to where the dataset is stored
  --dataset-kwargs DATASET_KWARGS
                        kew word arguments to be passed to dataset
                        constructor, should be given as a json object
  --model-tag MODEL_TAG
                        A tag to use for the model for saving results under
                        save-dir, defaults to the model arch and dataset used
  --save-dir SAVE_DIR   The path to the directory for saving results
  --dataset-parallel-calls DATASET_PARALLEL_CALLS
                        the number of parallel workers for dataset loading
  --shuffle-buffer-size SHUFFLE_BUFFER_SIZE
                        Shuffle buffer size for dataset loading
  --recipe-path RECIPE_PATH
                        The path to the yaml file containing the modifiers and
                        schedule to apply them with. If set to
                        'transfer_learning', then will create a schedule to
                        enable sparse transfer learning
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


##########
export command help:
usage: classification.py export [-h] --arch-key ARCH_KEY
                                [--pretrained PRETRAINED]
                                [--pretrained-dataset PRETRAINED_DATASET]
                                [--checkpoint-path CHECKPOINT_PATH]
                                [--model-kwargs MODEL_KWARGS] --dataset
                                DATASET --dataset-path DATASET_PATH
                                [--dataset-kwargs DATASET_KWARGS]
                                [--model-tag MODEL_TAG] [--save-dir SAVE_DIR]
                                [--num-samples NUM_SAMPLES]
                                [--onnx-opset ONNX_OPSET]

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
                        kew word arguments to be passed to model constructor,
                        should be given as a json object
  --dataset DATASET     The dataset to use for training, ex: imagenet,
                        imagenette, cifar10, etc. Set to imagefolder for a
                        generic dataset setup with an image folder structure
                        setup like imagenet or loadable by a dataset in
                        sparseml.tensorflow_v1.datasets
  --dataset-path DATASET_PATH
                        The root path to where the dataset is stored
  --dataset-kwargs DATASET_KWARGS
                        kew word arguments to be passed to dataset
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


##########
pruning_sensitivity command help:
usage: classification.py pruning_sensitivity [-h] --arch-key ARCH_KEY
                                             [--pretrained PRETRAINED]
                                             [--pretrained-dataset PRETRAINED_DATASET]
                                             [--checkpoint-path CHECKPOINT_PATH]
                                             [--model-kwargs MODEL_KWARGS]
                                             --dataset DATASET --dataset-path
                                             DATASET_PATH
                                             [--dataset-kwargs DATASET_KWARGS]
                                             [--model-tag MODEL_TAG]
                                             [--save-dir SAVE_DIR]
                                             [--dataset-parallel-calls
                                                DATASET_PARALLEL_CALLS]
                                             [--shuffle-buffer-size SHUFFLE_BUFFER_SIZE]
                                             [--approximate]
                                             [--steps-per-measurement
                                                STEPS_PER_MEASUREMENT]
                                             [--batch-size BATCH_SIZE]

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
                        kew word arguments to be passed to model constructor,
                        should be given as a json object
  --dataset DATASET     The dataset to use for training, ex: imagenet,
                        imagenette, cifar10, etc. Set to imagefolder for a
                        generic dataset setup with an image folder structure
                        setup like imagenet or loadable by a dataset in
                        sparseml.tensorflow_v1.datasets
  --dataset-path DATASET_PATH
                        The root path to where the dataset is stored
  --dataset-kwargs DATASET_KWARGS
                        kew word arguments to be passed to dataset
                        constructor, should be given as a json object
  --model-tag MODEL_TAG
                        A tag to use for the model for saving results under
                        save-dir, defaults to the model arch and dataset used
  --save-dir SAVE_DIR   The path to the directory for saving results
  --dataset-parallel-calls DATASET_PARALLEL_CALLS
                        the number of parallel workers for dataset loading
  --shuffle-buffer-size SHUFFLE_BUFFER_SIZE
                        Shuffle buffer size for dataset loading
  --approximate         True to approximate without running data through the
                        model, otherwise will run a one shot analysis
  --steps-per-measurement STEPS_PER_MEASUREMENT
                        The number of steps (batches) to run for each
                        measurement
  --batch-size BATCH_SIZE
                        The batch size to use while performing analysis


#########
EXAMPLES
#########

##########
Example command for pruning resnet50 on imagenet dataset:
python scripts/tensorflow_v1/classification.py train \
    --recipe-path ~/sparseml_recipes/pruning_resnet50.yaml \
    --arch-key resnet50 --dataset imagenet --dataset-path ~/datasets/ILSVRC2012 \
    --train-batch-size 256 --test-batch-size 1024

##########
Example command for transfer learning sparse mobilenet_v1 on an image folder dataset:
python scripts/tensorflow_v1/classification.py train \
    --sparse-transfer-learn \
    --recipe-path  ~/sparseml_recipes/pruning_mobilenet.yaml \
    --arch-key mobilenet_v1 --pretrained optim \
    --dataset imagefolder --dataset-path ~/datasets/my_imagefolder_dataset \
    --train-batch-size 256 --test-batch-size 1024

##########
Example command for exporting ResNet50:
python scripts/tensorflow_v1/classification.py export \
    --arch-key resnet50 --dataset imagenet --dataset-path ~/datasets/ILSVRC2012

##########
Example command for running approximated KS sensitivity analysis on mobilenet:
python scripts/tensorflow_v1/classification.py pruning_sensitivity \
    --approximate \
    --arch-key mobilenet --dataset imagenet \
    --dataset-path ~/datasets/ILSVRC2012

##########
Example command for running one shot KS sensitivity analysis on resnet50 for coco:
python scripts/tensorflow_v1/classification.py pruning_sensitivity \
    --arch-key resnet50 --dataset imagenet \
    --dataset-path ~/datasets/ILSVRC2012
"""

import argparse
import json
import math
import os
from typing import Dict, Optional, Tuple

import numpy

from sparseml import get_main_logger
from sparseml.tensorflow_v1.datasets import (
    Dataset,
    DatasetRegistry,
    create_split_iterators_handle,
)
from sparseml.tensorflow_v1.models import ModelRegistry
from sparseml.tensorflow_v1.optim import (
    ConstantPruningModifier,
    ScheduledModifierManager,
    pruning_loss_sens_magnitude,
    pruning_loss_sens_one_shot,
    pruning_loss_sens_op_vars,
)
from sparseml.tensorflow_v1.utils import (
    GraphExporter,
    accuracy,
    batch_cross_entropy_loss,
    tf_compat,
    write_simple_summary,
)
from sparseml.utils import create_dirs


LOGGER = get_main_logger()
TRAIN_COMMAND = "train"
EXPORT_COMMAND = "export"
PRUNING_SENSITVITY_COMMAND = "pruning_sensitivity"


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
    export_parser = subparsers.add_parser(
        EXPORT_COMMAND,
        description="Export a model to onnx as well as "
        "store sample inputs, outputs, and labels",
    )
    pruning_sensitivity_parser = subparsers.add_parser(
        PRUNING_SENSITVITY_COMMAND,
        description="Run a kernel sparsity (pruning) analysis for a given model",
    )

    parsers = [
        train_parser,
        export_parser,
        pruning_sensitivity_parser,
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
            "dataset in sparseml.tensorflow_v1.datasets",
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
            default="tensorflow_v1_classification",
            help="The path to the directory for saving results",
        )

        # task specific arguments
        if par in [train_parser, pruning_sensitivity_parser]:
            par.add_argument(
                "--dataset-parallel-calls",
                type=int,
                default=4,
                help="the number of parallel workers for dataset loading",
            )
            par.add_argument(
                "--shuffle-buffer-size",
                type=int,
                default=1000,
                help="Shuffle buffer size for dataset loading",
            )

        if par == train_parser:
            par.add_argument(
                "--recipe-path",
                type=str,
                default=None,
                help="The path to the yaml file containing the modifiers and "
                "schedule to apply them with. If set to 'transfer_learning', "
                "then will create a schedule to enable sparse transfer learning",
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
                "--logs-dir",
                type=str,
                default=os.path.join(
                    "tensorflow_v1_classification_train", "tensorboard-logs"
                ),
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
                "--init-lr",
                type=float,
                default=1e-9,
                help="The initial learning rate to use while training, "
                "the actual initial value used should be set by the sparseml recipe",
            )
            par.add_argument(
                "--optim-args",
                type=json.loads,
                default={},
                help="Additional args to be passed to the optimizer passed in"
                " as a json object",
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

        if par == pruning_sensitivity_parser:
            par.add_argument(
                "--approximate",
                action="store_true",
                help="True to approximate without running data through the model, "
                "otherwise will run a one shot analysis",
            )
            par.add_argument(
                "--steps-per-measurement",
                type=int,
                default=15,
                help="The number of steps (batches) to run for each measurement",
            )
            par.add_argument(
                "--batch-size",
                type=int,
                default=64,
                help="The batch size to use while performing analysis",
            )

    return parser.parse_args()


def _setup_save_dirs(args) -> Tuple[str, Optional[str]]:
    # logging and saving setup
    save_dir = os.path.abspath(os.path.expanduser(args.save_dir))
    logs_dir = (
        os.path.abspath(os.path.expanduser(os.path.join(args.logs_dir)))
        if args.command == TRAIN_COMMAND
        else None
    )

    if not args.model_tag:
        model_tag = "{}_{}".format(args.arch_key.replace("/", "."), args.dataset)
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

    # logs dir setup
    if args.command == TRAIN_COMMAND:
        logs_dir = os.path.join(logs_dir, model_id)
        create_dirs(logs_dir)
    else:
        logs_dir = None
    LOGGER.info("Model id is set to {}".format(model_id))
    return save_dir, logs_dir


def _create_dataset(args, train=True, image_size=None) -> Tuple[Dataset, int]:
    kwargs = args.dataset_kwargs
    if "image_size" in kwargs:
        image_size = kwargs["image_size"]
        del kwargs["image_size"]

    dataset = DatasetRegistry.create(
        args.dataset,
        root=args.dataset_path,
        train=train,
        image_size=image_size,
        **kwargs,
    )
    LOGGER.info("created {} dataset: {}".format("train" if train else "val", dataset))

    # get num_classes
    if args.dataset == "imagefolder":
        num_classes = dataset.num_classes
    else:
        dataset_attributes = DatasetRegistry.attributes(args.dataset)
        num_classes = dataset_attributes["num_classes"]

    return dataset, num_classes


def _build_dataset(args, dataset: Dataset, batch_size: int) -> Dataset:
    return dataset.build(
        batch_size,
        shuffle_buffer_size=args.shuffle_buffer_size,
        prefetch_buffer_size=batch_size,
        num_parallel_calls=args.dataset_parallel_calls,
    )


def _create_model(args, num_classes, inputs, training=False):
    outputs = ModelRegistry.create(
        args.arch_key,
        inputs,
        training=training,
        num_classes=num_classes,
        **args.model_kwargs,
    )
    LOGGER.info("created model {}".format(args.arch_key))
    return outputs


def _load_model(args, sess, checkpoint_path=None):
    sess.run(
        [
            tf_compat.global_variables_initializer(),
            tf_compat.local_variables_initializer(),
        ]
    )
    checkpoint_path = checkpoint_path or args.checkpoint_path
    ModelRegistry.load_pretrained(
        args.arch_key,
        pretrained=args.pretrained,
        pretrained_dataset=args.pretrained_dataset,
        pretrained_path=checkpoint_path,
        sess=sess,
    )
    if checkpoint_path:
        LOGGER.info("Loaded model weights from checkpoint: {}".format(checkpoint_path))


def _save_checkpoint(args, sess, save_dir, checkpoint_name) -> str:
    checkpoint_path = os.path.join(os.path.join(save_dir, checkpoint_name, "model"))
    create_dirs(checkpoint_path)
    saver = ModelRegistry.saver(args.arch_key)
    saved_name = saver.save(sess, checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, saved_name)
    LOGGER.info("Checkpoint saved to {}".format(checkpoint_path))
    return checkpoint_path


def _save_recipe(
    recipe_manager: ScheduledModifierManager,
    save_dir: str,
):

    recipe_save_path = os.path.join(save_dir, "recipe.yaml")
    recipe_manager.save(recipe_save_path)
    LOGGER.info(f"Saved recipe to {recipe_save_path}")


def train(args, save_dir, logs_dir):
    # setup dataset
    with tf_compat.device("/cpu:0"):
        train_dataset, _ = _create_dataset(args, train=True)
        val_dataset, num_classes = _create_dataset(args, train=False)
        # calc steps
        train_steps = math.ceil(len(train_dataset) / args.train_batch_size)
        val_steps = math.ceil(len(val_dataset) / args.test_batch_size)
        # build datasets
        train_dataset = _build_dataset(args, train_dataset, args.train_batch_size)
        val_dataset = _build_dataset(args, val_dataset, args.test_batch_size)
    handle, iterator, (train_iter, val_iter) = create_split_iterators_handle(
        [train_dataset, val_dataset]
    )

    # set up model graph
    images, labels = iterator.get_next()
    training = tf_compat.placeholder(dtype=tf_compat.bool, shape=[])
    outputs = _create_model(args, num_classes, images, training)

    # set up training objects
    loss = batch_cross_entropy_loss(outputs, labels)
    acc = accuracy(outputs, labels)
    global_step = tf_compat.train.get_or_create_global_step()
    train_op = tf_compat.train.AdamOptimizer(
        learning_rate=args.init_lr, **args.optim_args
    ).minimize(loss, global_step=global_step)
    update_ops = tf_compat.get_collection(tf_compat.GraphKeys.UPDATE_OPS)
    LOGGER.info("Created update ops for training")

    # set up sparseml modifier ops
    add_mods = (
        ConstantPruningModifier(params="__ALL__")
        if args.sparse_transfer_learn
        else None
    )
    manager = ScheduledModifierManager.from_yaml(
        file_path=args.recipe_path, add_modifiers=add_mods
    )
    mod_ops, mod_extras = manager.create_ops(train_steps, global_step)
    _save_recipe(recipe_manager=manager, save_dir=save_dir)
    with tf_compat.Session() as sess:
        # set up tensorboard logging
        summary_writer = tf_compat.summary.FileWriter(logs_dir, sess.graph)
        summaries = tf_compat.summary.merge_all()
        LOGGER.info("Logging to tensorboard at {}".format(logs_dir))

        # initialize variables, load pretrained weights, initialize modifiers
        train_iter_handle, val_iter_handle = sess.run(
            [train_iter.string_handle(), val_iter.string_handle()]
        )
        LOGGER.info("Initialized graph variables")
        _load_model(args, sess)
        manager.initialize_session()
        LOGGER.info("Initialized SparseML modifiers")

        best_loss = None
        for epoch in range(manager.max_epochs):
            # train
            LOGGER.info("Training for epoch {}...".format(epoch))
            sess.run(train_iter.initializer)
            train_acc, train_loss = [], []
            for step in range(train_steps):
                _, __, meas_step, meas_loss, meas_acc, meas_summ = sess.run(
                    [train_op, update_ops, global_step, loss, acc, summaries],
                    feed_dict={handle: train_iter_handle, training: True},
                )
                if step >= train_steps - 1:
                    # log the general summaries on the last training step
                    summary_writer.add_summary(meas_summ, meas_step)
                # run modifier ops
                sess.run(mod_ops)
                # summarize
                write_simple_summary(summary_writer, "Train/Loss", meas_loss, meas_step)
                write_simple_summary(
                    summary_writer, "Train/Acc", meas_acc * 100.0, meas_step
                )
                train_acc.append(meas_acc)
                train_loss.append(meas_loss)
            LOGGER.info(
                "Epoch {} - Train Loss: {}, Train Acc: {}".format(
                    epoch, numpy.mean(train_loss).item(), numpy.mean(train_acc).item()
                )
            )

            # val
            LOGGER.info("Validating for epoch {}...".format(epoch))
            sess.run(val_iter.initializer)
            val_acc, val_loss = [], []
            for step in range(val_steps):
                meas_loss, meas_acc = sess.run(
                    [loss, acc],
                    feed_dict={handle: val_iter_handle, training: False},
                )
                val_acc.append(meas_acc)
                val_loss.append(meas_loss)
                write_simple_summary(
                    summary_writer, "Val/Loss", numpy.mean(val_loss).item(), epoch
                )
                write_simple_summary(
                    summary_writer, "Val/Acc", numpy.mean(val_acc).item(), epoch
                )
            val_loss = numpy.mean(val_loss).item()
            LOGGER.info(
                "Epoch {} - Val Loss: {}, Val Acc: {}".format(
                    epoch, val_loss, numpy.mean(train_acc).item()
                )
            )
            if epoch >= args.save_best_after and (
                best_loss is None or val_loss <= best_loss
            ):
                _save_checkpoint(args, sess, save_dir, "checkpoint-best")
                best_loss = val_loss
            if args.save_epochs and epoch in args.save_epochs:
                _save_checkpoint(
                    args, sess, save_dir, "checkpoint-epoch-{}".format(epoch)
                )

        # cleanup graph and save final checkpoint
        manager.complete_graph()
        checkpoint_path = _save_checkpoint(args, sess, save_dir, "final-checkpoint")
    LOGGER.info("Running ONNX export flow")
    export(
        args,
        save_dir,
        checkpoint_path=checkpoint_path,
        skip_samples=True,
        num_classes=num_classes,
        opset=11,
    )


def export(
    args,
    save_dir,
    checkpoint_path=None,
    skip_samples=False,
    num_classes=None,
    opset=None,
):
    assert not skip_samples or num_classes
    # dataset creation
    if not skip_samples:
        val_dataset, num_classes = _create_dataset(args, train=False)

    with tf_compat.Graph().as_default():
        input_shape = ModelRegistry.input_shape(args.arch_key)
        inputs = tf_compat.placeholder(
            tf_compat.float32, [None] + list(input_shape), name="inputs"
        )
        outputs = _create_model(args, num_classes, inputs)

        with tf_compat.Session() as sess:
            _load_model(
                args, sess, checkpoint_path=checkpoint_path or args.checkpoint_path
            )

            exporter = GraphExporter(save_dir)

            if not skip_samples:
                # Export a batch of samples and expected outputs
                tf_dataset = val_dataset.build(
                    args.num_samples, repeat_count=1, num_parallel_calls=1
                )
                tf_iter = tf_compat.data.make_one_shot_iterator(tf_dataset)
                features, _ = tf_iter.get_next()
                inputs_val = sess.run(features)
                exporter.export_samples([inputs], [inputs_val], [outputs], sess)

            # Export model to tensorflow checkpoint format
            LOGGER.info("exporting tensorflow in {}".format(save_dir))
            exporter.export_checkpoint(sess=sess)

            # Export model to pb format
            LOGGER.info("exporting pb in {}".format(exporter.pb_path))
            exporter.export_pb(outputs=[outputs])

    # Export model to onnx format
    LOGGER.info("exporting onnx in {}".format(exporter.onnx_path))
    exporter.export_onnx([inputs], [outputs], opset=opset or args.onnx_opset)


def pruning_loss_sensitivity(args, save_dir):
    input_shape = ModelRegistry.input_shape(args.arch_key)
    train_dataset, num_classes = _create_dataset(
        args, train=True, image_size=input_shape[1]
    )
    with tf_compat.Graph().as_default() as graph:
        # create model graph
        inputs = tf_compat.placeholder(
            tf_compat.float32, [None] + list(input_shape), name="inputs"
        )
        outputs = _create_model(args, num_classes, inputs)

        with tf_compat.Session() as sess:
            _load_model(args, sess, checkpoint_path=args.checkpoint_path)
            if args.approximate:
                LOGGER.info("Running weight magnitude loss sensitivity analysis...")
                analysis = pruning_loss_sens_magnitude(graph, sess)
            else:
                op_vars = pruning_loss_sens_op_vars(graph)
                train_steps = math.ceil(len(train_dataset) / args.batch_size)
                train_dataset = _build_dataset(args, train_dataset, args.batch_size)
                handle, iterator, dataset_iter = create_split_iterators_handle(
                    [train_dataset]
                )
                dataset_iter = dataset_iter[0]
                images, labels = iterator.get_next()
                loss = batch_cross_entropy_loss(outputs, labels)
                tensor_names = ["inputs:0", labels.name]
                sess.run(dataset_iter.initializer)

                def feed_dict_creator(step: int) -> Dict[str, tf_compat.Tensor]:
                    assert step < train_steps
                    batch_data = [
                        tens.eval(session=sess) for tens in dataset_iter.get_next()
                    ]
                    return dict(zip(tensor_names, batch_data))

                LOGGER.info("Running one shot loss sensitivity analysis...")
                analysis = pruning_loss_sens_one_shot(
                    op_vars=op_vars,
                    loss_tensor=loss,
                    steps_per_measurement=args.steps_per_measurement,
                    feed_dict_creator=feed_dict_creator,
                    sess=sess,
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


def main(args):
    # set up saving and logging dirs
    save_dir, logs_dir = _setup_save_dirs(args)

    # RUN COMMAND SPECIFIC TASTS
    if args.command == TRAIN_COMMAND:
        train(args, save_dir, logs_dir)
    if args.command == EXPORT_COMMAND:
        export(args, save_dir)
    if args.command == PRUNING_SENSITVITY_COMMAND:
        pruning_loss_sensitivity(args, save_dir)


if __name__ == "__main__":
    args_ = parse_args()

    main(args_)
