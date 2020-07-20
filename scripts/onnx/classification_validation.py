"""
Script to validate a dataset's validation metrics for a given onnx model.
Uses neuralmagicML.pytorch for convenience functions to load images and metrics.
Additionally uses Neural Magic Inference Engine for inference of the model if available.


##########
Command help:
usage: classification_validation.py [-h] {neuralmagic,onnxruntime} ...

Evaluate an onnx model through Neural Magic or ONNXRuntime on a classification
dataset. Uses PyTorch datasets to load data.

positional arguments:
  {neuralmagic,onnxruntime}

optional arguments:
  -h, --help            show this help message and exit


##########
neuralmagic command help:
usage: classification_validation.py neuralmagic [-h] --onnx-file-path
                                                ONNX_FILE_PATH
                                                [--num-cores NUM_CORES]
                                                --dataset DATASET
                                                --dataset-path DATASET_PATH
                                                [--batch-size BATCH_SIZE]
                                                [--image-size IMAGE_SIZE]
                                                [--loader-num-workers LOADER_NUM_WORKERS]

Run validation in the Neural Magic inference engine

optional arguments:
  -h, --help            show this help message and exit
  --onnx-file-path ONNX_FILE_PATH
                        Path to the local onnx file to run validation for
  --num-cores NUM_CORES
                        Number of cores to use the Neural Magic engine with,
                        if left unset will use all detectable cores
  --dataset DATASET     The dataset to load for validation, ex: imagenet,
                        imagenette, cifar10, etc. Set to imagefolder for a
                        generic image classification dataset setup with an
                        image folder structure setup like imagenet.
  --dataset-path DATASET_PATH
                        The root path to where the dataset is stored
  --batch-size BATCH_SIZE
                        The batch size for the data to use to pass into the
                        model
  --image-size IMAGE_SIZE
                        The image size to use to pass into the model
  --loader-num-workers LOADER_NUM_WORKERS
                        The number of workers to use for data loading


##########
onnxruntime command help
usage: classification_validation.py onnxruntime [-h] --onnx-file-path
                                                ONNX_FILE_PATH --dataset
                                                DATASET --dataset-path
                                                DATASET_PATH
                                                [--batch-size BATCH_SIZE]
                                                [--image-size IMAGE_SIZE]
                                                [--loader-num-workers LOADER_NUM_WORKERS]

Run validation in onnxruntime

optional arguments:
  -h, --help            show this help message and exit
  --onnx-file-path ONNX_FILE_PATH
                        Path to the local onnx file to run validation for
  --dataset DATASET     The dataset to load for validation, ex: imagenet,
                        imagenette, cifar10, etc. Set to imagefolder for a
                        generic image classification dataset setup with an
                        image folder structure setup like imagenet.
  --dataset-path DATASET_PATH
                        The root path to where the dataset is stored
  --batch-size BATCH_SIZE
                        The batch size for the data to use to pass into the
                        model
  --image-size IMAGE_SIZE
                        The image size to use to pass into the model
  --loader-num-workers LOADER_NUM_WORKERS
                        The number of workers to use for data loading


##########
Example for ResNet, MobileNet, etc (image size 224) in neuralmagic:
python scripts/onnx/classification_validation.py neuralmagic  \
    --onnx-file-path /PATH/TO/model.onnx \
    --dataset imagenet \
    --dataset-path /PATH/TO/imagenet \
    --loader-num-workers 10 \
    --image-size 224


##########
Example for Inception V3 (image size 299) in onnxruntime:
python nm_imagenet_validation.py \
    python scripts/onnx/classification_validation.py neuralmagic  \
    --onnx-file-path /PATH/TO/model.onnx \
    --dataset imagenet \
    --dataset-path /PATH/TO/imagenet \
    --loader-num-workers 10 \
    --image-size 224 \
    --batch-size 1
"""

import argparse
from tqdm import auto

import torch
from torch.utils.data import DataLoader

from neuralmagicML import get_main_logger
from neuralmagicML.onnx.utils import ORTModelRunner, NMModelRunner
from neuralmagicML.pytorch.datasets import DatasetRegistry
from neuralmagicML.pytorch.utils import (
    CrossEntropyLossWrapper,
    TopKAccuracy,
    ModuleRunResults,
)


LOGGER = get_main_logger()
NEURALMAGIC_COMMAND = "neuralmagic"
ONNXRUNTIME_COMMAND = "onnxruntime"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an onnx model through Neural Magic or ONNXRuntime on a "
        "classification dataset. Uses PyTorch datasets to load data."
    )

    subparsers = parser.add_subparsers(dest="command")

    neuralmagic_parser = subparsers.add_parser(
        NEURALMAGIC_COMMAND,
        description="Run validation in the Neural Magic inference engine",
    )
    onnxruntime_parser = subparsers.add_parser(
        ONNXRUNTIME_COMMAND, description="Run validation in onnxruntime",
    )

    for index, par in enumerate([neuralmagic_parser, onnxruntime_parser]):
        # model args
        par.add_argument(
            "--onnx-file-path",
            type=str,
            required=True,
            help="Path to the local onnx file to run validation for",
        )

        if index == 0:
            par.add_argument(
                "--num-cores",
                type=int,
                default=-1,
                help="Number of cores to use the Neural Magic engine with, "
                "if left unset will use all detectable cores",
            )

        # dataset args
        par.add_argument(
            "--dataset",
            type=str,
            required=True,
            help="The dataset to load for validation, "
            "ex: imagenet, imagenette, cifar10, etc. "
            "Set to imagefolder for a generic image classification dataset setup "
            "with an image folder structure setup like imagenet.",
        )
        par.add_argument(
            "--dataset-path",
            type=str,
            required=True,
            help="The root path to where the dataset is stored",
        )
        par.add_argument(
            "--batch-size",
            type=int,
            default=16,
            help="The batch size for the data to use to pass into the model",
        )
        par.add_argument(
            "--image-size",
            type=int,
            default=224,
            help="The image size to use to pass into the model",
        )
        par.add_argument(
            "--loader-num-workers",
            type=int,
            default=4,
            help="The number of workers to use for data loading",
        )

    return parser.parse_args()


def main(args):
    # dataset creation
    LOGGER.info("Creating dataset...")
    val_dataset = DatasetRegistry.create(
        key=args.dataset,
        root=args.dataset_path,
        train=False,
        rand_trans=False,
        image_size=args.image_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_num_workers,
    )
    LOGGER.info("created val_dataset: {}".format(val_dataset))

    if args.command == NEURALMAGIC_COMMAND:
        LOGGER.info("creating model in neural magic...")
        runner = NMModelRunner(args.onnx_file_path, args.batch_size, args.num_cores)
    elif args.command == ONNXRUNTIME_COMMAND:
        LOGGER.info("creating model in onnxruntime...")
        runner = ORTModelRunner(args.onnx_file_path)
    else:
        raise ValueError("Unknown command given of {}".format(args.command))

    LOGGER.info("created runner: {}".format(runner))

    # val loss setup
    extras = {"top1acc": TopKAccuracy(1), "top5acc": TopKAccuracy(5)}
    val_loss = CrossEntropyLossWrapper(extras)
    LOGGER.info("created loss for validation: {}".format(val_loss))

    # calculate loss and accuracy
    res = ModuleRunResults()

    for batch, data in auto.tqdm(
        enumerate(val_loader), desc="Validation samples", total=len(val_loader)
    ):
        batch_x = {"input": data[0].numpy()}
        batch_size = data[0].shape[0]

        if batch_size != args.batch_size:
            LOGGER.warning(
                (
                    "skipping batch {} because it is not of expected batch size {}, "
                    "given {}"
                ).format(batch, args.batch_size, batch_size)
            )
            continue

        pred, pred_time = runner.batch_forward(batch_x)
        pred_pth = [torch.from_numpy(val) for val in pred.values()]
        batch_loss = val_loss(data, pred_pth)
        res.append(batch_loss, pred_pth[0].shape[0])

    # print out results instead of log so they can't be filtered
    print("\n\n{} validation results:".format(args.dataset))

    for loss_key in res.results.keys():
        print("\t{}: {}".format(loss_key, res.result_mean(loss_key)))


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
