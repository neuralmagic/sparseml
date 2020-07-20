"""
Image classification ks (pruning) sensitivity script.
Setup to support the following use cases:
- approximated loss sensitivity without running any data through a trained model
- one shot loss sensitivity without training the model at all while running
  data through it

Saves the results to a given directory.
Additionally will print the results out to the command line


##########
Command help:
usage: classification_sensitivity_ks.py [-h] [--approximate]
                                        [--steps-per-measurement STEPS_PER_MEASUREMENT]
                                        --arch-key ARCH_KEY
                                        [--pretrained PRETRAINED]
                                        [--pretrained-dataset PRETRAINED_DATASET]
                                        [--checkpoint-path CHECKPOINT_PATH]
                                        [--class-type CLASS_TYPE]
                                        [--device DEVICE] --dataset DATASET
                                        --dataset-path DATASET_PATH
                                        [--batch-size BATCH_SIZE]
                                        [--loader-num-workers LOADER_NUM_WORKERS]
                                        [--loader-pin-memory LOADER_PIN_MEMORY]
                                        [--model-tag MODEL_TAG]
                                        [--save-dir SAVE_DIR]

Run a kernel sparsity (pruning) analysis for a given model

optional arguments:
  -h, --help            show this help message and exit
  --approximate         True to approximate without running data through the
                        model, otherwise will run a one shot analysis
  --steps-per-measurement STEPS_PER_MEASUREMENT
                        The number of steps (batches) to run for each sparse
                        measurement
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
  --device DEVICE       The device to run on (can also include ids for data
                        parallel), ex: cpu, cuda, cuda:0,1
  --dataset DATASET     The dataset to use for training, ex: imagenet,
                        imagenette, cifar10, etc. Set to imagefolder for a
                        generic image classification dataset setup with an
                        image folder structure setup like imagenet.
  --dataset-path DATASET_PATH
                        The root path to where the dataset is stored
  --batch-size BATCH_SIZE
                        The batch size to use while training
  --loader-num-workers LOADER_NUM_WORKERS
                        The number of workers to use for data loading
  --loader-pin-memory LOADER_PIN_MEMORY
                        Use pinned memory for data loading
  --model-tag MODEL_TAG
                        A tag to use for the model for saving results under
                        save-dir and in tensorboard, defaults to the model
                        arch and dataset used
  --save-dir SAVE_DIR   The path to the directory for saving results


##########
Example command for running approximated KS sensitivity analysis on mobilenet:
python scripts/pytorch/classification_sensitivity_ks.py \
    --approximate --arch-key mobilenet --dataset imagenet \
    --dataset-path ~/datasets/ILSVRC2012


##########
Example command for running one shot KS sensitivity analysis on mobilenet for imagenet:
python scripts/pytorch/classification_sensitivity_ks.py \
    --arch-key mobilenet --dataset imagenet \
    --dataset-path ~/datasets/ILSVRC2012
"""

import argparse
import os

from torch.utils.data import DataLoader
import torch.nn.functional as TF

from neuralmagicML import get_main_logger
from neuralmagicML.pytorch.datasets import DatasetRegistry
from neuralmagicML.pytorch.models import ModelRegistry
from neuralmagicML.pytorch.utils import (
    LossWrapper,
    TopKAccuracy,
    model_to_device,
    default_device,
    PythonLogger,
)
from neuralmagicML.pytorch.recal import (
    approx_ks_loss_sensitivity,
    one_shot_ks_loss_sensitivity,
)
from neuralmagicML.utils import create_dirs


LOGGER = get_main_logger()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a kernel sparsity (pruning) analysis for a given model",
    )

    parser.add_argument(
        "--approximate",
        action="store_true",
        help="True to approximate without running data through the model, "
        "otherwise will run a one shot analysis",
    )
    parser.add_argument(
        "--steps-per-measurement",
        type=int,
        default=15,
        help="The number of steps (batches) to run for each sparse measurement",
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

    # training and dataset args
    parser.add_argument(
        "--device",
        type=str,
        default=default_device(),
        help="The device to run on (can also include ids for data parallel), ex: "
        "cpu, cuda, cuda:0,1",
    )
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="The batch size to use while training",
    )
    parser.add_argument(
        "--loader-num-workers",
        type=int,
        default=4,
        help="The number of workers to use for data loading",
    )
    parser.add_argument(
        "--loader-pin-memory",
        type=bool,
        default=True,
        help="Use pinned memory for data loading",
    )

    # logging and saving
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
        default="pytorch_classification_sensitivity_ks",
        help="The path to the directory for saving results",
    )

    return parser.parse_args()


def main(args):
    # logging and saving setup
    save_dir = os.path.abspath(os.path.expanduser(args.save_dir))

    model_tag = (
        "{}_{}".format(args.arch_key.replace("/", "."), args.dataset)
        if not args.model_tag
        else args.model_tag
    )
    model_id = model_tag
    model_inc = 0

    while os.path.exists(os.path.join(save_dir, model_id)):
        model_inc += 1
        model_id = "{}__{:02d}".format(model_tag, model_inc)

    save_dir = os.path.join(save_dir, model_id)
    create_dirs(save_dir)

    # loggers setup
    LOGGER.info("Model id is set to {}".format(model_id))

    # dataset creation
    if not args.approximate:
        input_shape = ModelRegistry.input_shape(args.arch_key)
        image_size = input_shape[1]  # assume shape [C, S, S] where S is the image size
        train_dataset = DatasetRegistry.create(
            args.dataset,
            root=args.dataset_path,
            train=True,
            rand_trans=True,
            image_size=image_size,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.loader_num_workers,
            pin_memory=args.loader_pin_memory,
        )
        LOGGER.info("created train_dataset: {}".format(train_dataset))
    else:
        train_dataset = None
        train_loader = None

    # model creation
    if args.dataset == "imagefolder":
        num_classes = train_dataset.num_classes
    else:
        dataset_attributes = DatasetRegistry.attributes(args.dataset)
        num_classes = dataset_attributes["num_classes"]

    model = ModelRegistry.create(
        args.arch_key,
        args.pretrained,
        args.checkpoint_path,
        args.pretrained_dataset,
        num_classes=num_classes,
        class_type=args.class_type,
    )
    LOGGER.info("created model: {}".format(model))

    # loss setup
    if not args.approximate:
        loss = LossWrapper(
            loss_fn=TF.cross_entropy,
            extras={"top1acc": TopKAccuracy(1), "top5acc": TopKAccuracy(5)},
        )
        LOGGER.info("created loss: {}".format(loss))
    else:
        loss = None

    # device setup
    if not args.approximate:
        module, device, device_ids = model_to_device(model, args.device)
    else:
        device = None
        device_ids = None

    # kernel sparsity analysis
    if args.approximate:
        analysis = approx_ks_loss_sensitivity(model)
    else:
        analysis = one_shot_ks_loss_sensitivity(
            model,
            train_loader,
            loss,
            device,
            args.steps_per_measurement,
            tester_loggers=[PythonLogger()],
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


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
