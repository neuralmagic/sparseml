"""
Image classification export script. Exports models to a standard structure
including an ONNX export as well as sample inputs, outputs, and labels


##########
Command help:
python scripts/pytorch/classification_export.py -h
usage: classification_export.py [-h] [--num-samples NUM_SAMPLES] --arch-key
                                ARCH_KEY [--pretrained PRETRAINED]
                                [--pretrained-dataset PRETRAINED_DATASET]
                                [--checkpoint-path CHECKPOINT_PATH]
                                [--class-type CLASS_TYPE] --dataset DATASET
                                --dataset-path DATASET_PATH
                                [--model-tag MODEL_TAG] [--save-dir SAVE_DIR]

Export an image classification model to onnx as well as store sample inputs,
outputs, and labels

optional arguments:
  -h, --help            show this help message and exit
  --num-samples NUM_SAMPLES
                        The number of samples to export along with the model
                        onnx and pth files (sample inputs and labels as well
                        as the outputs from model execution)
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
  --model-tag MODEL_TAG
                        A tag to use for the model for saving results under
                        save-dir, defaults to the model arch and dataset used
  --save-dir SAVE_DIR   The path to the directory for saving results


##########
Example command for exporting ResNet50:
python scripts/pytorch/classification_export.py \
    --arch-key resnet50 --dataset imagenet --dataset-path ~/datasets/ILSVRC2012
"""

import argparse
import os
from tqdm import auto

from torch.utils.data import DataLoader

from neuralmagicML.pytorch.datasets import DatasetRegistry
from neuralmagicML.pytorch.models import ModelRegistry
from neuralmagicML.pytorch.utils import (
    PythonLogger,
    ModuleExporter,
    early_stop_data_loader,
)
from neuralmagicML.utils import create_dirs


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

    return parser.parse_args()


def main(args):
    # logging and saving setup
    save_dir = os.path.abspath(os.path.expanduser(args.save_dir))

    if not args.model_tag:
        model_tag = "{}_{}".format(args.arch_key.replace("/", "."), args.dataset)
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
    py_logger = PythonLogger()
    py_logger.info("Model id is set to {}".format(model_id))

    # dataset creation
    input_shape = ModelRegistry.input_shape(args.arch_key)
    image_size = input_shape[1]  # assume shape [C, S, S] where S is the image size
    val_dataset = DatasetRegistry.create(
        args.dataset,
        root=args.dataset_path,
        train=False,
        rand_trans=False,
        image_size=image_size,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    val_loader = early_stop_data_loader(
        val_loader, args.num_samples if args.num_samples > 1 else 1
    )
    py_logger.info("created val_dataset: {}".format(val_dataset))

    # model creation
    if args.dataset == "imagefolder":
        num_classes = val_dataset.num_classes
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
    py_logger.info("created model: {}".format(model))

    # exporting
    exporter = ModuleExporter(model, save_dir)

    py_logger.info("exporting pytorch in {}".format(save_dir))
    exporter.export_pytorch()
    onnx_exported = False

    for batch, data in auto.tqdm(
        enumerate(val_loader),
        desc="Exporting samples",
        total=args.num_samples if args.num_samples > 1 else 1,
    ):
        if not onnx_exported:
            py_logger.info("exporting onnx in {}".format(save_dir))
            exporter.export_onnx(data[0])
            onnx_exported = True

        if args.num_samples > 0:
            exporter.export_samples(
                sample_batches=[data[0]], sample_labels=[data[1]], exp_counter=batch
            )


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
