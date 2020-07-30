"""
Torchvision models classification export script. Exports models to a standard structure
including an ONNX export as well as sample inputs, outputs, and labels.
Information about torchvision can be found here:
https://pytorch.org/docs/stable/torchvision/models.html


##########
Command help:
python scripts/pytorch/torchvision_export.py -h
usage: torchvision_export.py [-h] [--num-samples NUM_SAMPLES] --model MODEL
                             [--image-size IMAGE_SIZE]
                             [--pretrained PRETRAINED]
                             [--pretrained-dataset PRETRAINED_DATASET]
                             [--checkpoint-path CHECKPOINT_PATH] --dataset
                             DATASET --dataset-path DATASET_PATH
                             [--model-tag MODEL_TAG] [--save-dir SAVE_DIR]
                             [--onnx-opset ONNX_OPSET]

Export an image classification model to onnx as well as store sample inputs,
outputs, and labels

optional arguments:
  -h, --help            show this help message and exit
  --num-samples NUM_SAMPLES
                        The number of samples to export along with the model
                        onnx and pth files (sample inputs and labels as well
                        as the outputs from model execution)
  --model MODEL         The torchvision model class to use, ex: inception_v3,
                        resnet50, mobilenet_v2 model name is fed directly to
                        torchvision.models, more information can be found here
                        https://pytorch.org/docs/stable/torchvision/models.htm
                        l
  --image-size IMAGE_SIZE
                        Size of image to use for model input. Default is 224
                        unless pytorch documentation specifies otherwise
  --pretrained PRETRAINED
                        Set True to use torchvisions pretrained weights, to
                        not set weights, set False. default is true.
  --pretrained-dataset PRETRAINED_DATASET
                        The dataset to load pretrained weights for if
                        pretrained is set. Default is None which will load the
                        default dataset for the architecture. Ex can be set to
                        imagenet, cifar10, etc
  --checkpoint-path CHECKPOINT_PATH
                        A path to a previous checkpoint to load the state from
                        and resume the state for. If provided, pretrained will
                        be ignored
  --dataset DATASET     The dataset to use for training, ex: imagenet,
                        imagenette, cifar10, etc. Set to imagefolder for a
                        generic image classification dataset setup with an
                        image folder structure setup like imagenet.
  --dataset-path DATASET_PATH
                        The root path to where the dataset is stored
  --model-tag MODEL_TAG
                        A tag to use for the model for saving results under
                        save-dir, defaults to the model arch and dataset used
  --save-dir SAVE_DIR   The path to the directory for saving results
  --onnx-opset ONNX_OPSET
                        The onnx opset to use for export. Default is 11


##########
Example command for exporting ResNet50:
python scripts/pytorch/torchvision_export.py \
    --model resnet50 --dataset imagenet --dataset-path ~/datasets/ILSVRC2012
"""

import argparse
import os
from tqdm import auto

from torch.utils.data import DataLoader
from torchvision import models

from neuralmagicML import get_main_logger
from neuralmagicML.pytorch.datasets import DatasetRegistry
from neuralmagicML.pytorch.models import ModelRegistry
from neuralmagicML.pytorch.utils import (
    ModuleExporter,
    early_stop_data_loader,
    load_model,
)
from neuralmagicML.utils import create_dirs


LOGGER = get_main_logger()

MODEL_IMAGE_SIZES = {
    "inception_v3": 299,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export an image classification model to onnx as well as "
        "store sample inputs, outputs, and labels"
    )

    # recal
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="The number of samples to export along with the model onnx and pth files "
        "(sample inputs and labels as well as the outputs from model execution)",
    )
    # model args
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The torchvision model class to use, ex: inception_v3, resnet50, mobilenet_v2 "
        "model name is fed directly to torchvision.models, more information can be found here "
        "https://pytorch.org/docs/stable/torchvision/models.html",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        required=False,
        default=None,
        help="Size of image to use for model input. Default is 224 unless pytorch documentation "
        "specifies otherwise",
    )
    parser.add_argument(
        "--pretrained",
        type=bool,
        default=True,
        help="Set True to use torchvisions pretrained weights,"
        " to not set weights, set False. default is true.",
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

    # dataset args
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

    # logging and saving
    parser.add_argument(
        "--model-tag",
        type=str,
        default=None,
        help="A tag to use for the model for saving results under save-dir, "
        "defaults to the model arch and dataset used",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="pytorch_classification_export",
        help="The path to the directory for saving results",
    )
    parser.add_argument(
        "--onnx-opset",
        type=int,
        default=11,
        help="The onnx opset to use for export. Default is 11",
    )

    args = parser.parse_args()
    if args.image_size is None:
        args.image_size = (
            MODEL_IMAGE_SIZES[args.model] if args.model in MODEL_IMAGE_SIZES else 224
        )
    return args


def _get_torchvision_model(name, num_classes, pretrained=True, checkpoint_path=None):
    model_constructor = models.__getattribute__(name)
    model = model_constructor(pretrained=pretrained, num_classes=num_classes)
    if checkpoint_path is not None:
        load_model(checkpoint_path, model)
    return model


def main(args):
    # logging and saving setup
    save_dir = os.path.abspath(os.path.expanduser(args.save_dir))

    if not args.model_tag:
        model_tag = "{}_{}".format(args.model.replace("/", "."), args.dataset)
        model_id = model_tag
        model_inc = 0

        while os.path.exists(os.path.join(args.save_dir, model_id)):
            model_inc += 1
            model_id = "{}__{:02d}".format(model_tag, model_inc)
    else:
        model_id = args.model_tag

    save_dir = os.path.join(save_dir, model_id)
    create_dirs(save_dir)

    # loggers setup
    LOGGER.info("Model id is set to {}".format(model_id))

    # dataset creation
    val_dataset = DatasetRegistry.create(
        args.dataset,
        root=args.dataset_path,
        train=False,
        rand_trans=False,
        image_size=args.image_size,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    val_loader = early_stop_data_loader(
        val_loader, args.num_samples if args.num_samples > 1 else 1
    )
    LOGGER.info("created val_dataset: {}".format(val_dataset))

    # model creation
    if args.dataset == "imagefolder":
        num_classes = val_dataset.num_classes
    else:
        dataset_attributes = DatasetRegistry.attributes(args.dataset)
        num_classes = dataset_attributes["num_classes"]

    model = _get_torchvision_model(
        args.model, num_classes, args.pretrained, args.checkpoint_path,
    )
    LOGGER.info("created model: {}".format(model))

    # exporting
    exporter = ModuleExporter(model, save_dir)

    LOGGER.info("exporting pytorch in {}".format(save_dir))
    exporter.export_pytorch()
    onnx_exported = False

    for batch, data in auto.tqdm(
        enumerate(val_loader),
        desc="Exporting samples",
        total=args.num_samples if args.num_samples > 1 else 1,
    ):
        if not onnx_exported:
            LOGGER.info("exporting onnx in {}".format(save_dir))
            exporter.export_onnx(data[0], opset=args.onnx_opset)
            onnx_exported = True

        if args.num_samples > 0:
            exporter.export_samples(
                sample_batches=[data[0]], sample_labels=[data[1]], exp_counter=batch
            )


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
