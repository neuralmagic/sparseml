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
#####
Command help:
usage: export.py [-h] --arch-key ARCH_KEY [--pretrained PRETRAINED]
                 [--pretrained-dataset PRETRAINED_DATASET]
                 [--model-kwargs MODEL_KWARGS] --dataset DATASET
                 --dataset-path DATASET_PATH [--dataset-kwargs DATASET_KWARGS]
                 [--model-tag MODEL_TAG] [--save-dir SAVE_DIR]
                 [--checkpoint-path CHECKPOINT_PATH]
                 [--num-samples NUM_SAMPLES] [--onnx-opset ONNX_OPSET]
                 [--use-zipfile-serialization-if-available
                 USE_ZIPFILE_SERIALIZATION_IF_AVAILABLE]

Utility script to export a model to onnx and also store sample inputs/outputs

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
                        Keyword arguments to be passed to dataset constructor,
                        should be given as a json object
  --model-tag MODEL_TAG
                        A tag to use for the model for saving results under
                        save-dir, defaults to the model arch and dataset used
  --save-dir SAVE_DIR   The path to the directory for saving results
  --checkpoint-path CHECKPOINT_PATH
                        A path to a previous checkpoint to load the state from
                        and resume the state for. If provided, pretrained will
                        be ignored. If using a SparseZoo recipe, can also
                        provide 'zoo' to load the base weights associated with
                        that recipe
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
Example command for exporting ResNet50:
python integrations/pytorch/export.py \
    --arch-key resnet50 --dataset imagenet --dataset-path ~/datasets/ILSVRC2012
"""
import argparse
from typing import Any, Optional, Tuple

from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from sparseml import get_main_logger
from sparseml.pytorch.models import ModelRegistry
from sparseml.pytorch.utils import ModuleExporter
from sparseml.utils import convert_to_bool


CURRENT_TASK = utils.Tasks.EXPORT
LOGGER = get_main_logger()


def export(
    args: argparse.Namespace, model: Module, val_loader: DataLoader, save_dir: str
) -> None:
    """
    Utility method to export the model and data

     :params args : A Namespace object containing at-least the following keys
        use_zipfile_serialization_if_available, num_samples
    :param model: loaded model architecture to export
    :param val_loader: A DataLoader for validation data
    :param save_dir: Directory to store checkpoints at during training process
    """
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


def export_setup(args_: argparse.Namespace) -> Tuple[Module, Optional[str], Any]:
    """
    Pre-export setup

    :param args_: A Namespace object conbtaining atleast the following keys
        is_main_process, arch_key, approximate, eval_mode, local_rank, dataset,
        dataset_path, dataset_kwargs, rank, train_batch_size, batch_size,
        loader_num_workers, loader_pin_memory, model_kwargs, checkpoint_path,
        pretrained, pretrained_dataset}
    """
    save_dir, loggers = utils.get_save_dir_and_loggers(args_, task=CURRENT_TASK)
    input_shape = ModelRegistry.input_shape(args_.arch_key)
    image_size = input_shape[1]  # assume shape [C, S, S] where S is the image size
    (
        train_dataset,
        train_loader,
        val_dataset,
        val_loader,
    ) = utils.get_train_and_validation_loaders(args_, image_size, task=CURRENT_TASK)

    # model creation
    num_classes = utils.infer_num_classes(args_, train_dataset, val_dataset)
    model = utils.create_model(args_, num_classes)
    return model, save_dir, val_loader


def parse_args() -> argparse.Namespace:
    """
    Utility function to add and parse export specific command-line args
    """
    parser = argparse.ArgumentParser(
        description="Utility script to export a model to onnx "
        "and also store sample inputs/outputs"
    )

    utils.add_universal_args(parser=parser)

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help=(
            "A path to a previous checkpoint to load the state from and "
            "resume the state for. If provided, pretrained will be ignored"
            ". If using a SparseZoo recipe, can also provide 'zoo' to load "
            "the base weights associated with that recipe"
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="The number of samples to export along with the model onnx "
        "and pth files (sample inputs and labels as well as the outputs "
        "from model execution)",
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

    utils.append_ddp_defaults_and_preprocessing_args(args=args)
    return args


def main():
    """
    Driver function
    """
    args_ = parse_args()
    utils.distributed_setup(args_.local_rank)
    model, save_dir, val_loader = export_setup(args_)
    export(args_, model, val_loader, save_dir)


if __name__ == "__main__":
    main()
