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
usage: export.py [-h] --arch-key ARCH_KEY --dataset DATASET
                 --dataset-path DATASET_PATH
                 [--checkpoint-path CHECKPOINT_PATH]
                 [--num-samples NUM_SAMPLES] [--onnx-opset ONNX_OPSET]
                 [--use-zipfile-serialization-if-available
                 USE_ZIPFILE_SERIALIZATION_IF_AVAILABLE]
                 [--pretrained PRETRAINED]
                 [--pretrained-dataset PRETRAINED_DATASET]
                 [--model-kwargs MODEL_KWARGS]
                 [--dataset-kwargs DATASET_KWARGS]
                 [--model-tag MODEL_TAG] [--save-dir SAVE_DIR]

Utility script to export a model to onnx and also store sample
inputs/outputs

optional arguments:
  -h, --help            show this help message and exit
  --arch-key ARCH_KEY   The type of model to use, ex: resnet50, vgg16,
                        mobilenet put as help to see the full list
                        (will raise an exception with the list)
  --dataset DATASET     The dataset to use for exporting, ex: imagenet,
                        imagenette, cifar10, etc. Set to imagefolder
                        for a generic dataset setup with an image
                        folder structure setup like imagenet or
                        loadable by a dataset in
                        sparseml.pytorch.datasets
  --dataset-path DATASET_PATH
                        The root path to where the dataset is stored
  --checkpoint-path CHECKPOINT_PATH
                        A path to a previous checkpoint to load the
                        state from and resume the state for. If
                        provided, pretrained will be ignored . If using
                        a SparseZoo recipe, can also provide 'zoo' to
                        load the base weights associated with that
                        recipe
  --num-samples NUM_SAMPLES
                        The number of samples to export along with the
                        model onnx and pth files (sample inputs and
                        labels as well as the outputs from model
                        execution)
  --onnx-opset ONNX_OPSET
                        The onnx opset to use for export. Default is 11
  --use-zipfile-serialization-if-available
                        USE_ZIPFILE_SERIALIZATION_IF_AVAILABLE
                        for torch >= 1.6.0 only exports the Module's
                        state dict using the new zipfile serialization.
                        Default is True, has no affect on lower torch
                        versions
  --pretrained PRETRAINED
                        The type of pretrained weights to use, default
                        is true to load the default pretrained weights
                        for the model. Otherwise should be set to the
                        desired weights type: [base, optim, optim-
                        perf]. To not load any weights set to one of
                        [none, false]
  --pretrained-dataset PRETRAINED_DATASET
                        The dataset to load pretrained weights for if
                        pretrained is set. Default is None which will
                        load the default dataset for the architecture.
                        Ex can be set to imagenet, cifar10, etc
  --model-kwargs MODEL_KWARGS
                        Keyword arguments to be passed to model
                        constructor, should be given as a json object
  --dataset-kwargs DATASET_KWARGS
                        Keyword arguments to be passed to dataset
                        constructor, should be given as a json object
  --model-tag MODEL_TAG
                        A tag to use for the model for saving results
                        under save-dir, defaults to the model arch and
                        dataset used
  --save-dir SAVE_DIR   The path to the directory for saving results
##########
Example command for exporting ResNet50:
sparseml.image_classification.export_onnx \
    --arch-key resnet50 --dataset imagenet \
    --dataset-path ~/datasets/ILSVRC2012 \
    --checkpoint-path ~/checkpoints/resnet50_checkpoint.pth
"""
import json
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from sparseml import get_main_logger
from sparseml.pytorch.image_classification.utils import NmArgumentParser, helpers
from sparseml.pytorch.models import ModelRegistry
from sparseml.pytorch.utils import ModuleExporter
from sparseml.utils import convert_to_bool


CURRENT_TASK = helpers.Tasks.EXPORT
LOGGER = get_main_logger()


@dataclass
class ExportArgs:
    """
    Represents the arguments we use in our PyTorch integration scripts for
    exporting tasks

    Using :class:`NmArgumentParser` we can turn this class into `argparse
    <https://docs.python.org/3/library/argparse.html#module-argparse>`__
    arguments that can be specified on the command line.
    :param arch_key: A str key representing the type of model to use,
        ex:resnet50.
    :param dataset: The dataset to use for analysis, ex imagenet, imagenette,
        etc; Set to `imagefolder` for a custom dataset.
    :param dataset_path: Root path to dataset location.
    :param checkpoint_path: A path to a previous checkpoint to load the state
        from and resume the state for; Also works with SparseZoo recipes;
        Set to zoo to automatically download and load weights associated with a
        recipe.
    :param num_samples: The number of samples to export along with the model
        onnx and pth files (sample inputs and labels as well as the outputs
        from model execution). Default is 100.
    :param onnx_opset: The onnx opset to use for export. Default is 11.
    :param use_zipfile_serialization_if_available: for torch >= 1.6.0 only
        exports the Module's state dict using the new zipfile serialization.
        Default is True, has no affect on lower torch versions.
    :param pretrained: The type of pretrained weights to use,
        default is true to load the default pretrained weights for the model.
        Otherwise should be set to the desired weights type: [base, optim,
        optim-perf]. To not load any weights set to one of [none, false].
    :param pretrained_dataset: The dataset to load pretrained weights for if
        pretrained is set. Default is None which will load the default
        dataset for the architecture. Ex can be set to imagenet, cifar10, etc.
    :param model_kwargs: Keyword arguments to be passed to model constructor,
        should be given as a json object.
    :param dataset_kwargs: Keyword arguments to be passed to dataset
        constructor, should be given as a json object.
    :param model_tag: A tag to use for the model for saving results under
        save-dir, defaults to the model arch and dataset used.
    :param save_dir: The path to the directory for saving results.
    """

    dataset: str = field(
        metadata={
            "help": "The dataset to use for exporting, "
            "ex: imagenet, imagenette, cifar10, etc. "
            "Set to imagefolder for a generic dataset setup "
            "with an image folder structure setup like imagenet or "
            "loadable by a dataset in sparseml.pytorch.datasets"
        }
    )

    dataset_path: str = field(
        metadata={
            "help": "The root path to where the dataset is stored",
        }
    )
    checkpoint_path: str = field(
        default=None,
        metadata={
            "help": "A path to a previous checkpoint to load the state from "
            "and resume the state for. If provided, pretrained will "
            "be ignored . If using a SparseZoo recipe, can also "
            "provide 'zoo' to load the base weights associated with "
            "that recipe"
        },
    )
    arch_key: Optional[str] = field(
        default=None,
        metadata={
            "help": "The type of model to use, ex: resnet50, vgg16, mobilenet "
            "put as help to see the full list (will raise an exception "
            "with the list)",
        },
    )

    num_samples: int = field(
        default=100,
        metadata={
            "help": "The number of samples to export along with the model onnx "
            "and pth files (sample inputs and labels as well as the "
            "outputs from model execution)"
        },
    )

    onnx_opset: int = field(
        default=11, metadata={"help": "The onnx opset to use for export. Default is 11"}
    )

    use_zipfile_serialization_if_available: convert_to_bool = field(
        default=True,
        metadata={
            "help": "for torch >= 1.6.0 only exports the Module's state dict "
            "using the new zipfile serialization. Default is True, "
            "has no affect on lower torch versions"
        },
    )

    pretrained: str = field(
        default=True,
        metadata={
            "help": "The type of pretrained weights to use, "
            "default is true to load the default pretrained weights for "
            "the model. Otherwise should be set to the desired weights "
            "type: [base, optim, optim-perf]. To not load any weights "
            "set to one of [none, false]"
        },
    )

    pretrained_dataset: str = field(
        default=None,
        metadata={
            "help": "The dataset to load pretrained weights for if pretrained "
            "is set. Default is None which will load the default "
            "dataset for the architecture. Ex can be set to imagenet, "
            "cifar10, etc",
        },
    )

    model_kwargs: json.loads = field(
        default_factory=lambda: {},
        metadata={
            "help": "Keyword arguments to be passed to model constructor, should "
            "be given as a json object"
        },
    )

    dataset_kwargs: json.loads = field(
        default_factory=lambda: {},
        metadata={
            "help": "Keyword arguments to be passed to dataset constructor, "
            "should be  given as a json object",
        },
    )

    model_tag: str = field(
        default=None,
        metadata={
            "help": "A tag to use for the model for saving results under save-dir, "
            "defaults to the model arch and dataset used",
        },
    )

    save_dir: str = field(
        default="pytorch_vision",
        metadata={
            "help": "The path to the directory for saving results",
        },
    )

    def __post_init__(self):
        self.arch_key = helpers.get_arch_key(
            arch_key=self.arch_key,
            checkpoint_path=self.checkpoint_path,
        )

        if "preprocessing_type" not in self.dataset_kwargs and (
            "coco" in self.dataset.lower() or "voc" in self.dataset.lower()
        ):
            if "ssd" in self.arch_key.lower():
                self.dataset_kwargs["preprocessing_type"] = "ssd"
            elif "yolo" in self.arch_key.lower():
                self.dataset_kwargs["preprocessing_type"] = "yolo"

        self.local_rank: int = -1
        self.is_main_process: bool = True


def export(
    args: ExportArgs, model: Module, val_loader: DataLoader, save_dir: str
) -> None:
    """
    Utility method to export the model and data

    :param args : An ExportArgs object containing config for export task.
    :param model: loaded model architecture to export
    :param val_loader: A DataLoader for validation data
    :param save_dir: Directory to store checkpoints at during exporting process
    """
    exporter = ModuleExporter(model, save_dir)

    # export PyTorch state dict
    LOGGER.info(f"exporting pytorch in {save_dir}")

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
            LOGGER.info(f"exporting onnx in {save_dir}")
            exporter.export_onnx(data[0], opset=args.onnx_opset, convert_qat=True)
            onnx_exported = True

        if args.num_samples > 0:
            exporter.export_samples(
                sample_batches=[data[0]], sample_labels=[data[1]], exp_counter=batch
            )


def export_setup(args_: ExportArgs) -> Tuple[Module, Optional[str], Any]:
    """
    Pre-export setup

    :param args_ : An ExportArgs object containing config for export task.
    """
    save_dir, loggers = helpers.get_save_dir_and_loggers(
        task=CURRENT_TASK,
        is_main_process=args_.is_main_process,
        save_dir=args_.save_dir,
        arch_key=args_.arch_key,
        model_tag=args_.model_tag,
        dataset_name=args_.dataset,
    )
    input_shape = ModelRegistry.input_shape(key=args_.arch_key)
    image_size = input_shape[1]  # assume shape [C, S, S] where S is the image size

    val_dataset, val_loader = helpers.get_dataset_and_dataloader(
        dataset_name=args_.dataset,
        dataset_path=args_.dataset_path,
        batch_size=1,
        image_size=image_size,
        dataset_kwargs=args_.dataset_kwargs,
        training=False,
        loader_num_workers=1,
        loader_pin_memory=False,
        max_samples=args_.num_samples,
    )

    train_dataset = None

    # model creation
    num_classes = helpers.infer_num_classes(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        dataset=args_.dataset,
        model_kwargs=args_.model_kwargs,
    )
    model, args_.arch_key = helpers.create_model(
        checkpoint_path=args_.checkpoint_path,
        recipe_path=None,
        num_classes=num_classes,
        arch_key=args_.arch_key,
        pretrained=args_.pretrained,
        pretrained_dataset=args_.pretrained_dataset,
        local_rank=args_.local_rank,
        **args_.model_kwargs,
    )
    return model, save_dir, val_loader


def main():
    """
    Driver function
    """
    _parser = NmArgumentParser(
        dataclass_types=ExportArgs,
        description="Utility script to export a model to onnx "
        "and also store sample inputs/outputs",
    )
    (args_,) = _parser.parse_args_into_dataclasses()
    model, save_dir, val_loader = export_setup(args_)
    export(args_, model, val_loader, save_dir)


if __name__ == "__main__":
    main()
