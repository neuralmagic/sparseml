"""
Model export script. Exports models to a standard structure
including an ONNX export as well as sample inputs, outputs, and labels


##########
Command help:
python scripts/pytorch/model_export.py -h
usage: model_export.py [-h] [--num-samples NUM_SAMPLES] --arch-key ARCH_KEY
                 [--pretrained PRETRAINED]
                 [--pretrained-dataset PRETRAINED_DATASET]
                 [--checkpoint-path CHECKPOINT_PATH]
                 [--model-kwargs MODEL_KWARGS] --dataset DATASET
                 --dataset-path DATASET_PATH [--model-tag MODEL_TAG]
                 [--save-dir SAVE_DIR] [--onnx-opset ONNX_OPSET]
                 [--use-zipfile-serialization-if-available USE_ZIPFILE_SERIALIZATION_IF_AVAILABLE]

Export a model to onnx as well as store sample inputs, outputs, and labels

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
  --model-kwargs MODEL_KWARGS
                        Comma separated string key word arguments for the
                        model constructor in the form of
                        key1=value1,key2=value2
  --dataset DATASET     The dataset to use for training, ex: imagenet,
                        imagenette, cifar10, etc. Set to imagefolder for a
                        generic dataset setup with an image folder structure
                        setup like imagenet or loadable by a dataset in
                        neuralmagicML.pytorch.datasets
  --dataset-path DATASET_PATH
                        The root path to where the dataset is stored
  --model-tag MODEL_TAG
                        A tag to use for the model for saving results under
                        save-dir, defaults to the model arch and dataset used
  --save-dir SAVE_DIR   The path to the directory for saving results
  --onnx-opset ONNX_OPSET
                        The onnx opset to use for export. Default is 11
  --use-zipfile-serialization-if-available USE_ZIPFILE_SERIALIZATION_IF_AVAILABLE
                        for torch >= 1.6.0 only exports the Module's state
                        dict using the new zipfile serialization. Default is
                        True, has no affect on lower torch versions


##########
Example command for exporting ResNet50:
python scripts/pytorch/model_export.py \
    --arch-key resnet50 --dataset imagenet --dataset-path ~/datasets/ILSVRC2012

Example command for exporting multiclass MobilenetV1 predictor
python scripts/pytorch/model_export.py \
    --arch-key mobilenet-v1 --dataset imagenet --dataset-path ~/datasets/ILSVRC2012 \
    --model-kwargs class_type=multi

Example command for exporting ssd300_resnet50:
python scripts/pytorch/model_export.py \
    --arch-key ssd300_resnet50 --dataset coco --dataset-path ~/data/coco-detection --pretrained false
"""

import argparse
import os
from tqdm import auto

from torch.utils.data import DataLoader

from neuralmagicML import get_main_logger
from neuralmagicML.pytorch.datasets import DatasetRegistry
from neuralmagicML.pytorch.models import ModelRegistry
from neuralmagicML.pytorch.utils import ModuleExporter, early_stop_data_loader
from neuralmagicML.utils import convert_to_bool, create_dirs


LOGGER = get_main_logger()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a model to onnx as well as "
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
        "--model-kwargs",
        type=str,
        default="",
        help="Comma separated string key word arguments for the model constructor "
        "in the form of key1=value1,key2=value2",
    )

    # dataset args
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The dataset to use for training, "
        "ex: imagenet, imagenette, cifar10, etc. "
        "Set to imagefolder for a generic dataset setup "
        "with an image folder structure setup like imagenet or loadable by a "
        "dataset in neuralmagicML.pytorch.datasets",
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
        default="pytorch_model_export",
        help="The path to the directory for saving results",
    )
    parser.add_argument(
        "--onnx-opset",
        type=int,
        default=11,
        help="The onnx opset to use for export. Default is 11",
    )
    parser.add_argument(
        "--use-zipfile-serialization-if-available",
        type=convert_to_bool,
        default=True,
        help="for torch >= 1.6.0 only exports the Module's state dict "
        "using the new zipfile serialization. Default is True, has no "
        "affect on lower torch versions",
    )

    args = parser.parse_args()
    args.model_kwargs = (
        dict(arg.strip().split("=") for arg in args.model_kwargs.strip().split(","))
        if args.model_kwargs
        else {}
    )

    return args


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
    LOGGER.info("Model id is set to {}".format(model_id))

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
    LOGGER.info("created val_dataset: {}".format(val_dataset))

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
        **args.model_kwargs,
    )
    LOGGER.info("created model: {}".format(model))

    # exporting
    exporter = ModuleExporter(model, save_dir)

    LOGGER.info("exporting pytorch in {}".format(save_dir))
    exporter.export_pytorch(
        use_zipfile_serialization_if_available=args.use_zipfile_serialization_if_available
    )
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
