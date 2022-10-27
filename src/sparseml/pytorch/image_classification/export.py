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
Usage: export.py [OPTIONS]

  SparseML-PyTorch Integration for exporting image classification models to
  onnx along with sample inputs and outputs

Options:
  --dataset TEXT                  The dataset used for training, ex:
                                  `imagenet`, `imagenette`, `cifar10`, etc.
                                  Set to `imagefolder` for a generic dataset
                                  setup with imagefolder type structure like
                                  imagenet or loadable by a dataset in
                                  `sparseml.pytorch.datasets`
  --dataset-path, --dataset_path DIRECTORY
                                  The root dir path where the dataset is
                                  stored or should be downloaded to if
                                  available
  --checkpoint-path, --checkpoint_path TEXT
                                  A path to a previous checkpoint to load the
                                  state from and resume the state for
                                  exporting
  --labels-to-class-mapping , --labels_to_class_mapping TEXT
                                  Optional path to the dataset-specific
                                  mapping from numeric labels to human-readable
                                  class strings. Expected to be a path to a
                                  .json file containing a serialized dictionary
                                  [default: None]
  --arch_key, --arch-key TEXT     The architecture key for image
                                  classification model; example: `resnet50`,
                                  `mobilenet`. Note: Will be read from the
                                  checkpoint if not specified
  --num-samples, --num_samples INTEGER
                                  The number of samples to export along with
                                  the model onnx and pth files (sample inputs
                                  and labels as well as the outputs from model
                                  execution)  [default: -1]
   --onnx-opset, --onnx_opset INTEGER
                                  The onnx opset to use for exporting the
                                  model  [default: 11]
  --use_zipfile_serialization_if_available, --use-zipfile-serialization-if-available /
  --no_zipfile_serialization, --no-zipfile-serialization
                                  For torch >= 1.6.0 only exports the Module's
                                  state dict using the new zipfile
                                  serialization. Default is True, has no
                                  effect with lower torch versions  [default:
                                  use_zipfile_serialization_if_available]
  --pretrained TEXT               The type of pretrained weights to use, loads
                                  default pretrained weights for the model if
                                  not specified or set to `True`. Otherwise
                                  should be set to the desired weights type:
                                  [base, optim, optim-perf]. To not load any
                                  weights set to one of [none, false]
                                  [default: True]
  --pretrained-dataset, --pretrained_dataset TEXT
                                  The dataset to load pretrained weights for
                                  if pretrained is set. Load the default
                                  dataset for the architecture if set to None.
                                  examples:`imagenet`, `cifar10`, etc...
  --model-kwargs, --model_kwargs TEXT
                                  Keyword arguments to be passed to model
                                  constructor, should be given as a json
                                  object
  --dataset-kwargs, --dataset_kwargs TEXT
                                  Keyword arguments to be passed to dataset
                                  constructor, should be specified as a json
                                  object
  --model-tag, --model_tag TEXT   A tag for saving results under save-dir,
                                  defaults to the model arch and dataset used
  --save-dir, --save_dir DIRECTORY
                                  The path to the directory for saving results
                                  [default: pytorch_vision]
  --image-size, --image_size INTEGER
                                  The size of the image input to the model.
                                  Value should be equal to S for [C, S, S] or
                                  [S, S, C] dimensional input  [default: 224]
  --recipe TEXT                   The path to a recipe file or SparseZoo stub
                                  to use for exporting the model
  --convert_qat, --convert-qat / --no_convert_qat, --no-convert-qat
                                  if True, exports of torch QAT graphs will be
                                  converted to a fully quantized
                                  representation. Default is True  [default:
                                  convert_qat]
  --num-classes, --num_classes INTEGER
                                  number of classes for model; must be set if
                                  a dataset is not provided
  --help                          Show this message and exit.


##########
Example command for exporting ResNet50:
sparseml.image_classification.export_onnx \
    --arch-key resnet50 --dataset imagenet \
    --dataset-path ~/datasets/ILSVRC2012 \
    --checkpoint-path ~/checkpoints/resnet50_checkpoint.pth
"""
import json
import os
from typing import Any, Dict, Optional, Union

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

import click
from sparseml import get_main_logger
from sparseml.pytorch.image_classification.utils import cli_helpers, helpers
from sparseml.pytorch.opset import TORCH_DEFAULT_ONNX_OPSET
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import ModuleExporter, load_model


CURRENT_TASK = helpers.Tasks.EXPORT
LOGGER = get_main_logger()


@click.command()
@click.option(
    "--dataset",
    type=str,
    default=None,
    help="The dataset used for training, "
    "ex: `imagenet`, `imagenette`, `cifar10`, etc. "
    "Set to `imagefolder` for a generic dataset setup with "
    "imagefolder type structure like imagenet or loadable by "
    "a dataset in `sparseml.pytorch.datasets`",
)
@click.option(
    "--dataset-path",
    "--dataset_path",
    type=click.Path(dir_okay=True, file_okay=False),
    callback=cli_helpers.create_dir_callback,
    default=None,
    help="The root dir path where the dataset is stored or should "
    "be downloaded to if available",
)
@click.option(
    "--checkpoint-path",
    "--checkpoint_path",
    type=str,
    default=None,
    help="A path to a previous checkpoint to load the state from "
    "and resume the state for exporting",
)
@click.option(
    "--labels-to-class-mapping",
    "--labels_to_class_mapping",
    type=str,
    default=None,
    help="Optional path to the dataset-specific mapping from "
    "numeric labels to human-readable class strings. "
    "Expected to be a path to a .json file containing "
    "a serialized dictionary",
)
@click.option(
    "--arch_key",
    "--arch-key",
    type=str,
    default=None,
    help="The architecture key for image classification model; "
    "example: `resnet50`, `mobilenet`. "
    "Note: Will be read from the checkpoint if not specified",
)
@click.option(
    "--num-samples",
    "--num_samples",
    type=int,
    default=-1,
    show_default=True,
    help="The number of samples to export along with the model onnx "
    "and pth files (sample inputs and labels as well as the "
    "outputs from model execution)",
)
@click.option(
    "--onnx-opset",
    "--onnx_opset",
    type=int,
    default=TORCH_DEFAULT_ONNX_OPSET,
    show_default=True,
    help="The onnx opset to use for exporting the model",
)
@click.option(
    "--use_zipfile_serialization_if_available/--no_zipfile_serialization",
    "--use-zipfile-serialization-if-available/--no-zipfile-serialization",
    is_flag=True,
    default=True,
    show_default=True,
    help="For torch >= 1.6.0 only exports the Module's state dict "
    "using the new zipfile serialization. Default is True, "
    "has no effect with lower torch versions",
)
@click.option(
    "--pretrained",
    type=str,
    default=True,
    show_default=True,
    help="The type of pretrained weights to use, "
    "loads default pretrained weights for "
    "the model if not specified or set to `True`. "
    "Otherwise should be set to the desired weights "
    "type: [base, optim, optim-perf]. To not load any weights set"
    " to one of [none, false]",
)
@click.option(
    "--pretrained-dataset",
    "--pretrained_dataset",
    type=str,
    default=None,
    show_default=True,
    help="The dataset to load pretrained weights for if pretrained is "
    "set. Load the default dataset for the architecture if set to None. "
    "examples:`imagenet`, `cifar10`, etc...",
)
@click.option(
    "--model-kwargs",
    "--model_kwargs",
    default=json.dumps({}),
    type=str,
    callback=cli_helpers.parse_json_callback,
    help="Keyword arguments to be passed to model constructor, should "
    "be given as a json object",
)
@click.option(
    "--dataset-kwargs",
    "--dataset_kwargs",
    default=json.dumps({}),
    type=str,
    callback=cli_helpers.parse_json_callback,
    help="Keyword arguments to be passed to dataset constructor, "
    "should be specified as a json object",
)
@click.option(
    "--model-tag",
    "--model_tag",
    type=str,
    default=None,
    help="A tag for saving results under save-dir, "
    "defaults to the model arch and dataset used",
)
@click.option(
    "--save-dir",
    "--save_dir",
    type=click.Path(dir_okay=True, file_okay=False),
    default="pytorch_vision",
    callback=cli_helpers.create_dir_callback,
    show_default=True,
    help="The path to the directory for saving results",
)
@click.option(
    "--image-size",
    "--image_size",
    type=int,
    default=224,
    show_default=True,
    help="The size of the image input to the model. Value should be "
    "equal to S for [C, S, S] or [S, S, C] dimensional input",
)
@click.option(
    "--recipe",
    type=str,
    default=None,
    help="The path to a recipe file or SparseZoo stub to use for exporting the model",
)
@click.option(
    "--convert_qat/--no_convert_qat",
    "--convert-qat/--no-convert-qat",
    default=True,
    show_default=True,
    help="if True, exports of torch QAT graphs will be converted to a fully quantized "
    "representation. Default is True",
)
@click.option(
    "--num-classes",
    "--num_classes",
    type=int,
    default=None,
    show_default=True,
    help="number of classes for model; must be set if a dataset is not provided",
)
def main(
    dataset: Optional[str] = None,
    dataset_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    labels_to_class_mapping: Optional[str] = None,
    arch_key: Optional[str] = None,
    num_samples: int = -1,
    onnx_opset: int = TORCH_DEFAULT_ONNX_OPSET,
    use_zipfile_serialization_if_available: bool = True,
    pretrained: Union[str, bool] = True,
    pretrained_dataset: Optional[str] = None,
    model_kwargs: Dict[str, Any] = {},
    dataset_kwargs: Dict[str, Any] = {},
    model_tag: Optional[str] = None,
    save_dir: str = "pytorch_vision",
    image_size: int = 224,
    recipe: Optional[str] = None,
    convert_qat: bool = True,
    num_classes: Optional[int] = None,
):
    """
    SparseML-PyTorch Integration for exporting image classification models to
    onnx along with sample inputs and outputs
    """

    os.makedirs(save_dir, exist_ok=True)
    _validate_dataset_num_classes(dataset, dataset_path, num_samples, num_classes)

    save_dir, loggers = helpers.get_save_dir_and_loggers(
        task=CURRENT_TASK,
        is_main_process=True,
        save_dir=save_dir,
        arch_key=arch_key,
        model_tag=model_tag,
        dataset_name=dataset or "",
    )

    if dataset:
        val_dataset, val_loader = helpers.get_dataset_and_dataloader(
            dataset_name=dataset,
            dataset_path=dataset_path,
            batch_size=1,
            image_size=image_size,
            dataset_kwargs=dataset_kwargs,
            training=False,
            loader_num_workers=1,
            loader_pin_memory=False,
            max_samples=num_samples,
        )
    else:
        val_dataset = None
        val_loader = None

    # model creation
    if num_classes is None:
        num_classes = helpers.infer_num_classes(
            train_dataset=None,
            val_dataset=val_dataset,
            dataset=dataset,
            model_kwargs=model_kwargs,
        )

    model, arch_key, _ = helpers.create_model(
        checkpoint_path=checkpoint_path if recipe is None else None,
        recipe_path=recipe,
        num_classes=num_classes,
        arch_key=arch_key,
        pretrained=pretrained,
        pretrained_dataset=pretrained_dataset,
        local_rank=-1,
        **model_kwargs,
    )

    if labels_to_class_mapping is None:
        labels_to_class_mapping = helpers.label_to_class_mapping_from_dataset(dataset)

    if recipe is not None:
        ScheduledModifierManager.from_yaml(recipe).apply_structure(model)
        load_model(checkpoint_path, model, strict=True)
    export(
        model=model,
        val_loader=val_loader,
        save_dir=save_dir,
        use_zipfile_serialization_if_available=use_zipfile_serialization_if_available,
        num_samples=num_samples,
        onnx_opset=onnx_opset,
        convert_qat=convert_qat,
        image_size=image_size,
        labels_to_class_mapping=labels_to_class_mapping,
    )


def export(
    model: Module,
    val_loader: "DataLoader",
    save_dir: str,
    use_zipfile_serialization_if_available: bool,
    num_samples: int,
    onnx_opset: int = TORCH_DEFAULT_ONNX_OPSET,
    convert_qat: bool = True,
    image_size: int = 224,
    labels_to_class_mapping: Optional[Union[str, Dict[int, str]]] = None,
) -> None:
    """
    Utility method to export the model and data

    :param model: loaded model architecture to export
    :param val_loader: A DataLoader for validation data
    :param save_dir: Directory to store checkpoints at during exporting process
    :param use_zipfile_serialization_if_available: Whether to use zipfile
        serialization during export
    :param num_samples: Number of samples to export
    :param onnx_opset: ONNX opset version to use
    :param convert_qat: set True to convert QAT export to fully quantized
        representation
    :param image_size: size of image to export
    :param labels_to_class_mapping: information about the mapping
        from integer labels to string class names. Can be either a string
        (path to the .json serialized dictionary) or a dictionary.
    """
    exporter = ModuleExporter(model, save_dir)

    if not val_loader:
        # create fake data for export
        val_loader = [[torch.randn(1, 3, image_size, image_size)]]

    onnx_exported = False

    for batch, data in tqdm(
        enumerate(val_loader),
        desc="Exporting samples",
        total=num_samples if num_samples > 1 else 1,
    ):
        if not onnx_exported:
            # export onnx file using first sample for graph freezing
            LOGGER.info(f"exporting onnx in {save_dir}")
            exporter.export_onnx(data[0], opset=onnx_opset, convert_qat=convert_qat)
            onnx_exported = True

        if num_samples > 0:
            exporter.export_samples(
                sample_batches=[data[0]], sample_labels=[data[1]], exp_counter=batch
            )

    exporter.create_deployment_folder(labels_to_class_mapping=labels_to_class_mapping)


def _validate_dataset_num_classes(
    dataset: str,
    dataset_path: str,
    num_samples: int,
    num_classes: int,
):
    if dataset and not dataset_path:
        raise ValueError(f"found dataset {dataset} but dataset_path not specified")
    if dataset_path and not dataset:
        raise ValueError(f"found dataset_path {dataset_path} but dataset not specified")
    if num_classes is None and (not dataset or not dataset_path):
        raise ValueError(
            "If num_classes is not provided, both dataset and dataset_path must be "
            "set to infer num_classes"
        )
    if num_samples > 0 and (not dataset or not dataset_path):
        raise ValueError(
            "If num_samples > 0, both dataset and dataset_path must be "
            "set to load samples for export"
        )


if __name__ == "__main__":
    main()
