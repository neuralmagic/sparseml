"""
Script to validate a dataset's validation metrics for a given onnx model.
Uses neuralmagicML.pytorch for convenience functions to load images and metrics.
Additionally uses Neural Magic Inference Engine for inference of the model if available.


##########
Command help:
python scripts/onnx/classification_validation.py -h
neuralmagic package not found in system, falling back to onnx runtime
usage: classification_validation.py [-h] --onnx-file-path ONNX_FILE_PATH
                                    [--num-cores NUM_CORES] --dataset DATASET
                                    --dataset-path DATASET_PATH
                                    [--batch-size BATCH_SIZE]
                                    [--image-size IMAGE_SIZE]
                                    [--loader-num-workers LOADER_NUM_WORKERS]

Evaluate an onnx model through Neural Magic (if not available will fall back
to onnx runtime) on a classification dataset. Uses PyTorch datasets to load
data.

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
Example for ResNet, MobileNet, etc (image size 224):
python nm_imagenet_validation.py \
    --onnx-file-path PATH/TO/MODEL.onnx \
    --num-cores 4 --dataset imagenet --dataset-path /PATH/TO/IMAGENET \
     --loader-num-workers 10 --image-size 224


##########
Example for Inception V3 (image size 299):
python nm_imagenet_validation.py \
    --onnx-file-path PATH/TO/MODEL.onnx \
    --num-cores 4 --dataset imagenet --dataset-path /PATH/TO/IMAGENET \
     --loader-num-workers 10 --image-size 299
"""

import argparse
import logging
from tqdm import auto
from onnxruntime import InferenceSession

import torch
from torch.utils.data import DataLoader

from neuralmagicML.onnx.utils import onnx_nodes_sparsities
from neuralmagicML.pytorch.datasets import DatasetRegistry
from neuralmagicML.pytorch.utils import (
    CrossEntropyLossWrapper,
    TopKAccuracy,
    ModuleRunResults,
    PythonLogger,
)

try:
    from neuralmagic import create_model
except Exception:
    logging.warning(
        "neuralmagic package not found in system, falling back to onnx runtime"
    )
    create_model = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an onnx model through Neural Magic "
        "(if not available will fall back to onnx runtime) on a "
        "classification dataset. Uses PyTorch datasets to load data."
    )

    # model args
    parser.add_argument(
        "--onnx-file-path",
        type=str,
        required=True,
        help="Path to the local onnx file to run validation for",
    )
    parser.add_argument(
        "--num-cores",
        type=int,
        default=-1,
        help="Number of cores to use the Neural Magic engine with, "
        "if left unset will use all detectable cores",
    )

    # dataset args
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The dataset to load for validation, "
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
        default=16,
        help="The batch size for the data to use to pass into the model",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="The image size to use to pass into the model",
    )
    parser.add_argument(
        "--loader-num-workers",
        type=int,
        default=4,
        help="The number of workers to use for data loading",
    )

    return parser.parse_args()


def validation_metrics(
    onnx_file_path: str,
    num_cores: int,
    image_size: int,
    dataset: str,
    dataset_path: str,
    test_batch_size: int,
    loader_num_workers: int,
    py_logger: PythonLogger,
) -> ModuleRunResults:
    """
    Calculate validation metrics for an onnx model over a given dataset

    :param onnx_file_path: path to the onnx file
    :param num_cores: number of cores to run the inference engine on,
        only applicable for neuralmagic package
    :param image_size: the size of the image to feed through the model
    :param dataset: the type of dataset to use; ex: imagenet
    :param dataset_path: path to the dataset,
        should be setup to work with the PyTorch version of the dataset
    :param test_batch_size: the batch size to run through the model for validation
    :param loader_num_workers: number of workers to use with the PyTorch
        data loader
    :param py_logger: the python logger instance to log updates to
    :return: the validation results after running over all of the validation dataset
    """
    # dataset creation
    val_dataset = DatasetRegistry.create(
        key=dataset,
        root=dataset_path,
        train=False,
        rand_trans=False,
        image_size=image_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=loader_num_workers,
    )
    py_logger.info("created val_dataset: {}".format(val_dataset))

    if create_model is not None:
        model = create_model(
            onnx_file_path, batch_size=test_batch_size, num_cores=num_cores
        )
        input_name = None
        py_logger.info("created model in neural magic: {}".format(model))
    else:
        model = InferenceSession(onnx_file_path)
        input_name = model.get_inputs()[0].name
        py_logger.info("created model in onnx runtime: {}".format(model))

    # val loss setup
    extras = {"top1acc": TopKAccuracy(1), "top5acc": TopKAccuracy(5)}
    val_loss = CrossEntropyLossWrapper(extras)
    py_logger.info("created loss for validation: {}".format(val_loss))

    # calculate loss and accuracy
    res = ModuleRunResults()

    for batch, data in auto.tqdm(
        enumerate(val_loader), desc="Validation samples", total=len(val_loader)
    ):
        batch_x = data[0].numpy()
        batch_size = batch_x.shape[0]

        if batch_size != test_batch_size:
            py_logger.info(
                (
                    "skipping batch {} because it is not of expected batch size {}, "
                    "given {}"
                ).format(batch, test_batch_size, batch_size)
            )
            continue

        if create_model is None:
            pred = model.run(None, {input_name: batch_x})
        else:
            pred = model.forward([batch_x])

        pred_pth = [torch.from_numpy(val) for val in pred]
        batch_loss = val_loss(data, pred_pth)
        res.append(batch_loss, pred_pth[0].shape[0])

    return res


def main(args):
    py_logger = PythonLogger()

    py_logger.info("calculating model sparsity")
    total_sparse, node_sparse = onnx_nodes_sparsities(args.onnx_file_path)

    py_logger.info("running {} validation metrics".format(args.dataset))
    res = validation_metrics(
        args.onnx_file_path,
        args.num_cores,
        args.image_size,
        args.dataset,
        args.dataset_path,
        args.batch_size,
        args.loader_num_workers,
        py_logger,
    )

    py_logger.info("node inp sparsities:")
    for name, val in node_sparse.items():
        py_logger.info("{}: {}".format(name, val))

    py_logger.info("\ntotal sparsity: {}".format(total_sparse))

    py_logger.info("\n\n{} validation results:".format(args.dataset))

    for loss_key in res.results.keys():
        py_logger.info("\t{}: {}".format(loss_key, res.result_mean(loss_key)))


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
