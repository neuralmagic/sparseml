"""
Image classification recal script. Setup to support the following use cases:
- training image classification architectures
- pruning image classification architectures
- transfer learning image classification architectures
- evaluating image classification architectures


##########
Command help:
usage: classification_train.py [-h] [--recal-config-path RECAL_CONFIG_PATH]
                               [--sparse-transfer-learn] [--eval-mode]
                               --arch-key ARCH_KEY [--pretrained PRETRAINED]
                               [--pretrained-dataset PRETRAINED_DATASET]
                               [--checkpoint-path CHECKPOINT_PATH]
                               [--class-type CLASS_TYPE]
                               --dataset DATASET --dataset-path DATASET_PATH
                               --train-batch-size TRAIN_BATCH_SIZE
                               --eval-batch-size TEST_BATCH_SIZE
                               [--eval_every_n_steps EVAL_STEPS]
                               [--save_ckpt_every_n_steps CHECKPOINT_STEPS]
                               [--optimizer OPTIM]
                               [--optimizer-params OPTIM_PARAMS]
                               [--model-tag MODEL_TAG] [--save-dir SAVE_DIR]
                               [--logs-dir LOGS_DIR]

Train and/or prune an image classification architecture on a dataset

optional arguments:
  -h, --help            show this help message and exit
  --recal-config-path RECAL_CONFIG_PATH
                        The path to the yaml file containing the modifiers and
                        schedule to apply them with. If set to
                        'transfer_learning', then will create a schedule to
                        enable sparse transfer learning
  --sparse-transfer-learn
                        Enable sparse transfer learning modifiers to enforce
                        the sparsity for already sparse layers. The modifiers
                        are added to the ones to be loaded from the recal-
                        config-path
  --eval-mode           Puts into evaluation mode so that the model can be
                        evaluated on the desired dataset
  --arch-key ARCH_KEY   The type of model to create, ex: resnet50, vgg16,
                        mobilenet put as help to see the full list (will raise
                        an exception with the list)
  --pretrained PRETRAINED
                        The type of pretrained weights to use, default is true
                        to load the default pretrained weights for the model.
                        Otherwise should be set to the desired weights type:
                        [base, recal, recal-perf]. To not load any weights set
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
  --class-type CLASS_TYPE
                        One of [single, multi] where single is for single
                        class training using a softmax and multi is for multi
                        class training using a sigmoid
  --dataset DATASET     The dataset to use for training, ex: imagenet,
                        imagenette, cifar10, etc. Set to imagefolder for a
                        generic image classification dataset setup with an
                        image folder structure setup like imagenet.
  --dataset-path DATASET_PATH
                        The root path to where the dataset is stored
  --train-batch-size TRAIN_BATCH_SIZE
                        The batch size to use while training
  --eval-batch-size TEST_BATCH_SIZE
                        The batch size to use while testing
  --optimizer OPTIM
                        The optimizer type to use, chosen from tf.compat.v1.train
                        (e.g., AdamOptimizer)
  --optimizer-params OPTIM_PARAMS
                        Additional args to be passed to the optimizer passed
                        in as a json dict
  --loss LOSS
                        Name of the loss function
  --num-epochs NUM_EPOCHS
                        Number of training epochs
  --train-build-config TRAIN_BUILD_CONFIG
                        Additional parameter dictionary to build the train TF dataset;
                        default to '{"shuffle_buffer_size": 100,
                        "prefetch_buffer_size": 512, "num_parallel_calls": 4}'
  --metrics METRICS
                        Metrics to collect during evaluation phase; must be a metric
                        defined in tf.compat.v1.metrics
  --eval-build-config CONFIG
                        Additional parameter dictionary to build the eval dataset;
                        default to '{"shuffle_buffer_size": 100,
                        "prefetch_buffer_size": 512, "num_parallel_calls": 4}'
  --eval-every-n-steps STEPS
                        Run evaluation after every such number of steps
  --save-ckpt-every-n-steps STEPS
                        Save checkpoint after every such number of steps
  --eval-session-name  EVAL_SESS_NAME
                        Name of the evaluation session

  --eval-checkpoint-path EVAL_CHECKPOINT_PATH
                        Path to a specific checkpoint to evaluate;
                        if None then the latest in model directory is used
  --model-tag MODEL_TAG
                        A tag to use for the model for saving results under
                        save-dir and in tensorboard, defaults to the model
                        arch and dataset used
  --save-dir SAVE_DIR   The path to the directory for saving results
  --logs-dir LOGS_DIR   The path to the directory for saving logs


##########
Example: training resnet20 on cifar10 dataset using the Adam optimizer

python scripts/tensorflow/classification_train.py \
    --arch-key resnet20 --dataset cifar10 \
    --dataset-path datasets/cifar10/ \
    --train-batch-size 128 --eval-batch-size 1000 \
    --optimizer AdamOptimizer --optimizer-params '{"learning_rate": 0.001}' \
    --num-epochs 200 \
    --save-dir resnet20_cifar10/

##########
Example: training/pruning resnet20 on cifar10 dataset using the Adam optimizer with
learning rate and pruning schedule defined in a recalibration config file

python scripts/tensorflow/classification_train.py \
    --arch-key resnet20 --dataset cifar10 \
    --dataset-path datasets/cifar10/ \
    --train-batch-size 128 --eval-batch-size 1000 \
    --optimizer AdamOptimizer \
    --recal-config-path configs/resnet20_cifar10_training.yaml \
    --num-epochs 200 \
    --save-dir resnet20_cifar10/

##########
Example: fine-tuning a trained model

python scripts/tensorflow/classification_train.py \
    --arch-key resnet20 --dataset cifar10 \
    --dataset-path datasets/cifar10/ \
    --checkpoint-path resnet20_cifar10/resnet20_cifar10/model.ckpt-1955
    --train-batch-size 128 --eval-batch-size 1000 \
    --num-epochs 50
    --save-dir resnet20_cifar10/


##########
Example: evaluating a resnet20 model on cifar10 given a checkpoint

python scripts/tensorflow/classification_train.py \
    --arch-key resnet20 --dataset cifar10 \
    --dataset-path datasets/cifar10/ \
    --eval-mode --eval-checkpoint-path resnet20_cifar10/model.ckpt-1955 \
    --eval-batch-size 1024

"""

import argparse
from typing import Tuple, Union
import logging
import os
import time
import json
import math

from neuralmagicML import get_main_logger
from neuralmagicML.tensorflow.datasets import DatasetRegistry
from neuralmagicML.tensorflow.models import ModelRegistry
from neuralmagicML.tensorflow.recal import (
    ScheduledModifierManager,
    ConstantKSModifier,
)
from neuralmagicML.tensorflow.utils import tf_compat
from neuralmagicML.utils import create_dirs

LOGGER = get_main_logger()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and/or prune an image classification "
        "architecture on a dataset"
    )

    parser.add_argument(
        "--tf-log",
        type=int,
        default=tf_compat.logging.INFO,
        help="Level of Tensorflow logging:  "
        "DEBUG = 10, ERROR = 40, FATAL = 50, INFO = 20 (default), WARN = 30",
    )

    # recal
    parser.add_argument(
        "--recal-config-path",
        type=str,
        default=None,
        help="The path to the yaml file containing the modifiers and "
        "schedule to apply them with. If set to 'transfer_learning', "
        "then will create a schedule to enable sparse transfer learning",
    )
    parser.add_argument(
        "--sparse-transfer-learn",
        action="store_true",
        help="Enable sparse transfer learning modifiers to enforce the sparsity "
        "for already sparse layers. The modifiers are added to the "
        "ones to be loaded from the recal-config-path",
    )
    parser.add_argument(
        "--eval-mode",
        action="store_true",
        help="Puts into evaluation mode so that the model can be "
        "evaluated on the desired dataset",
    )

    # model args
    parser.add_argument(
        "--arch-key",
        type=str,
        required=True,
        help="The type of model to create, ex: resnet50, vgg16, mobilenet "
        "put as help to see the full list (will raise an exception with the list)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=True,
        help="The type of pretrained weights to use, "
        "default is true to load the default pretrained weights for the model. "
        "Otherwise should be set to the desired weights type: "
        "[base, recal, recal-perf]. "
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
        "--checkpoint-path",
        type=str,
        default=None,
        help="A path to a previous checkpoint to load the state from and "
        "resume the state for. If provided, pretrained will be ignored",
    )
    parser.add_argument(
        "--class-type",
        type=str,
        default="single",
        help="One of [single, multi] where single is for single class training "
        "using a softmax and multi is for multi class training using a sigmoid",
    )

    ############################
    # Options for datasets
    ############################
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The dataset to use for training, "
        "ex: imagenet, imagenette, cifar10, etc. "
        "Set to imagefolder for a generic image classification dataset setup "
        "with an image folder structure setup like imagenet.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="The root path to where the dataset is stored",
    )

    ############################
    # Options for training
    ############################
    parser.add_argument(
        "--optimizer", type=str, default="AdamOptimizer", help="Name of the optimizer"
    )

    # Disable black formating since json accepts only single quote for the
    # default dictionary string
    parser.add_argument(
        "--optimizer-params",
        type=json.loads,
        default='{"learning_rate": 0.001}',
        help="Additional parameters of the optimizer",
    )

    parser.add_argument(
        "--loss", type=str, default="cross_entropy", help="Name of the loss function"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=None, help="Number of training epochs"
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="Batch size used for training",
    )

    parser.add_argument(
        "--train-build-config",
        type=json.loads,
        default='{"shuffle_buffer_size": 100, "prefetch_buffer_size": 512, "num_parallel_calls": 4}',
        help="Additional parameter dictionary to build the train dataset",
    )

    #################################
    # Options for evaluation
    #################################
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Batch size used for evaluation",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["accuracy"],
        help="Metrics to collect during evaluation phase",
    )

    parser.add_argument(
        "--eval-build-config",
        type=json.loads,
        default='{"repeat_count": 1, "prefetch_buffer_size": 512, "num_parallel_calls": 4}',
        help="Additional parameter dictionary to build the eval dataset",
    )

    # Evaluation options also applied in the train and evaluation mode
    parser.add_argument(
        "--eval-every-n-steps", type=int, default=100, help="Run eval every N steps"
    )
    parser.add_argument(
        "--save-ckpt-every-n-steps",
        type=int,
        default=100,
        help="Save checkpoint every N steps",
    )

    # Evaluation options applied without training involved
    parser.add_argument(
        "--eval-session-name",
        type=str,
        default="default_session",
        help="Name of the evaluation session",
    )
    parser.add_argument(
        "--eval-checkpoint-path",
        type=str,
        default=None,
        help="Path to a specific checkpoint to evaluate; if None then the latest in model directory is used.",
    )

    #############################
    # logging and saving
    #############################
    parser.add_argument(
        "--model-tag",
        type=str,
        default=None,
        help="A tag to use for the model for saving results under save-dir "
        "and in tensorboard, defaults to the model arch and dataset used",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="tensorflow_classification_train",
        help="The path to the directory for saving results",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default=os.path.join("tensorflow_classification_train", "tensorboard-logs"),
        help="The path to the directory for saving logs",
    )

    return parser.parse_args()


def populate_model_fn_params(num_examples, logs_dir, args):
    """
    Create parameters for model function

    :param logs_dir: directory for logging
    :param args: commandline input arguments
    :return: dictionary of parameters
    """
    batch_size = args.eval_batch_size if args.eval_mode else args.train_batch_size
    steps_per_epoch = int(num_examples / batch_size)
    base_name_scope = ModelRegistry._ATTRIBUTES[args.arch_key].base_name_scope
    model_fn_params = {
        "arch_key": args.arch_key,
        "class_type": args.class_type,
        "loss": args.loss,
        "metrics": args.metrics,
        "optimizer": args.optimizer,
        "optimizer_params": args.optimizer_params,
        "steps_per_epoch": steps_per_epoch,
        "logs_dir": logs_dir,
        "eval_every_n_steps": args.eval_every_n_steps,
        "recal_config_path": args.recal_config_path,
        "sparse_transfer_learn": args.sparse_transfer_learn,
        "pretrained": args.pretrained,
        "checkpoint_path": args.checkpoint_path,
        "eval_checkpoint_path": args.eval_checkpoint_path,
        "pretrained_dataset": args.pretrained_dataset,
        "eval_mode": args.eval_mode,
        "base_name_scope": base_name_scope,
    }
    return model_fn_params


def populate_run_config(args):
    """
    Create parameters for run config, used for creating estimator

    :param args: commandline input arguments
    :return: a RunConfig instance
    """
    run_config = tf_compat.estimator.RunConfig(
        save_summary_steps=args.eval_every_n_steps,
        save_checkpoints_steps=args.save_ckpt_every_n_steps,
    )
    return run_config


def main(args):

    tf_compat.logging.set_verbosity(args.tf_log)

    # logging and saving setup
    save_dir = os.path.abspath(os.path.expanduser(args.save_dir))
    logs_dir = os.path.abspath(os.path.expanduser(os.path.join(args.logs_dir)))

    if not args.model_tag:
        model_tag = "{}_{}".format(args.arch_key.replace("/", "."), args.dataset)
        model_id = model_tag
        model_inc = 0

        while os.path.exists(os.path.join(logs_dir, model_id)):
            model_inc += 1
            model_id = "{}__{:02d}".format(model_tag, model_inc)
    else:
        model_id = args.model_tag

    save_dir = os.path.join(save_dir, model_id)
    logs_dir = os.path.join(logs_dir, model_id)
    create_dirs(save_dir)
    create_dirs(logs_dir)

    LOGGER.info("Model id is set to {}".format(model_id))

    train = not args.eval_mode
    dataset = DatasetRegistry.create(args.dataset, root=args.dataset_path, train=train)
    num_examples_per_epoch = len(dataset)
    LOGGER.info("Number of examples per epoch: {}".format(num_examples_per_epoch))

    if args.dataset == "imagefolder":
        num_classes = dataset.num_classes
    else:
        dataset_attributes = DatasetRegistry.attributes(args.dataset)
        num_classes = dataset_attributes["num_classes"]

    model_fn_params = populate_model_fn_params(num_examples_per_epoch, logs_dir, args)
    run_config = populate_run_config(args)

    classifier = ModelRegistry.create_estimator(
        args.arch_key,
        save_dir,
        model_fn_params,
        run_config,
        num_classes=num_classes,
        class_type=args.class_type,
        training=train,
    )
    if args.eval_mode:
        # Evaluation mode
        input_fn = dataset.build_input_fn(
            args.eval_batch_size, **args.eval_build_config
        )
        metrics = classifier.evaluate(
            input_fn=input_fn,
            steps=math.ceil(num_examples_per_epoch / args.eval_batch_size),
            checkpoint_path=args.eval_checkpoint_path,
            name=args.eval_session_name,
        )
        LOGGER.info("Evaluation metrics: {}".format(metrics))
    else:
        # Training mode
        steps_per_epoch = math.ceil(num_examples_per_epoch / args.train_batch_size)
        max_steps = steps_per_epoch * args.num_epochs
        train_input_fn = dataset.build_input_fn(
            args.train_batch_size, **args.train_build_config
        )
        train_spec = tf_compat.estimator.TrainSpec(
            input_fn=train_input_fn, max_steps=max_steps
        )

        val_dataset = DatasetRegistry.create(
            args.dataset, root=args.dataset_path, train=False
        )
        num_eval_examples = len(val_dataset)
        eval_input_fn = val_dataset.build_input_fn(
            args.eval_batch_size, **args.eval_build_config
        )
        eval_spec = tf_compat.estimator.EvalSpec(
            input_fn=eval_input_fn,
            steps=math.ceil(num_eval_examples / args.eval_batch_size),
            throttle_secs=1,
        )
        tf_compat.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    # add sleep to make sure all background processes have finished,
    # ex tensorboard writing
    time.sleep(5)


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
