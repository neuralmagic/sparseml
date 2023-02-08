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
import deeplake
from torchvision import transforms as deeplake_transforms
import json
import torch
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import torchvision
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

import click
from sparseml.pytorch.models.registry import ModelRegistry
from sparseml.pytorch.opset import TORCH_DEFAULT_ONNX_OPSET
from sparseml.pytorch.optim.manager import ScheduledModifierManager
from sparseml.pytorch.torchvision import presets
from sparseml.pytorch.utils import ModuleExporter
from sparseml.pytorch.utils.model import load_model


_LOGGER = logging.getLogger(__name__)


@click.command(
    context_settings=dict(
        token_normalize_func=lambda x: x.replace("-", "_"), show_default=True
    )
)
@click.option(
    "--arch-key",
    type=str,
    required=True,
    help="The architecture key for image classification model; "
    "example: `resnet50`, `mobilenet`. ",
)
@click.option(
    "--checkpoint-path",
    type=str,
    required=True,
    help="A path to a previous checkpoint to load the state from "
    "and resume the state for exporting, or a zoo stub.",
)
@click.option(
    "--one-shot",
    default=None,
    type=str,
    help="Path to recipe to use to apply in a one-shot manner",
)
@click.option(
    "--labels-to-class-mapping",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, path_type=Path),
    default=None,
    help="Optional path to the dataset-specific mapping from "
    "numeric labels to human-readable class strings. "
    "Expected to be a path to a .json file containing "
    "a serialized dictionary",
)
@click.option(
    "--num-samples",
    type=int,
    default=-1,
    help="The number of samples to export along with the model onnx "
    "and pth files (sample inputs and labels as well as the "
    "outputs from model execution)",
)
@click.option(
    "--onnx-opset",
    type=int,
    default=TORCH_DEFAULT_ONNX_OPSET,
    help="The onnx opset to use for exporting the model",
)
@click.option(
    "--save-dir",
    type=click.Path(dir_okay=True, file_okay=False, path_type=Path),
    default="torchvision",
    help="The path to the directory for saving results",
)
@click.option(
    "--convert-qat/--no-convert-qat",
    default=True,
    is_flag=True,
    help="if True, exports of torch QAT graphs will be converted to a fully quantized "
    "representation.",
)
@click.option(
    "--interpolation",
    default="bilinear",
    type=str,
    help="the interpolation method",
)
@click.option(
    "--img-resize-size",
    default=256,
    type=int,
    help="the resize size used for validation",
)
@click.option(
    "--img-crop-size",
    default=224,
    type=int,
    help="the central crop size used for validation",
)
@click.option(
    "--deeplake_data_url", default=None, type=str, help="deeplake train dataset url"
)
@click.option(
    "--deeplake_image_column",
    default="images",
    type=str,
    help="Image column of the dataset",
)
@click.option(
    "--deeplake_label_column",
    default="labels",
    type=str,
    help="Label column of the dataset",
)
@click.option(
    "--deeplake_token", default=None, type=str, help="Token to authenticate download"
)
def main(
    arch_key: str,
    checkpoint_path: str,
    one_shot: Optional[str],
    labels_to_class_mapping: Optional[Path],
    num_samples: int,
    onnx_opset: int,
    save_dir: Path,
    convert_qat: bool,
    interpolation: str,
    img_resize_size: int,
    img_crop_size: int,
    deeplake_data_url: Optional[str],
    deeplake_token: Optional[str],
    deeplake_label_column: Optional[str],
    deeplake_image_column: Optional[str],
):
    """
    SparseML torchvision integration for exporting image classification models to
    onnx along with sample inputs and outputs
    """

    save_dir.mkdir(parents=True, exist_ok=True)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    resize_size = 256
    interpolation = InterpolationMode.BILINEAR

    tform = deeplake_transforms.Compose(
        [
            deeplake_transforms.RandomRotation(20),  # Image augmentation
            deeplake_transforms.Resize(resize_size, interpolation=interpolation),
            deeplake_transforms.ToTensor(),  # Must convert to pytorch tensor for subsequent operations to run
            deeplake_transforms.Normalize(mean=mean, std=std),
        ]
    )
    ds_train = deeplake.load(path=deeplake_data_url, token=deeplake_token)
    num_classes = len(ds_train[deeplake_label_column].info.class_names)
    data_loader = ds_train.pytorch(
        tensors=[deeplake_image_column, deeplake_label_column],
        return_index=False,
        num_workers=0,
        shuffle=False,
        transform={
            deeplake_image_column: tform,
            deeplake_label_column: torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(torch.tensor),
                    torchvision.transforms.Lambda(torch.squeeze),
                ]
            ),
        },
        batch_size=1,
        decode_method={deeplake_image_column: "pil"},
    )

    if arch_key in ModelRegistry.available_keys():
        model = ModelRegistry.create(key=arch_key, num_classes=num_classes)
    elif arch_key in torchvision.models.__dict__:
        # fall back to torchvision
        model = torchvision.models.__dict__[arch_key](
            pretrained=False, num_classes=num_classes
        )
    else:
        raise ValueError(
            f"Unable to find {arch_key} in ModelRegistry or in torchvision.models"
        )

    load_model(checkpoint_path, model, strict=True)

    if one_shot is not None:
        ScheduledModifierManager.from_yaml(one_shot).apply(model)

    if labels_to_class_mapping is not None:
        with open(labels_to_class_mapping) as fp:
            labels_to_class_mapping = json.load(fp)

    export(
        model=model,
        val_loader=data_loader,
        save_dir=save_dir,
        num_samples=num_samples,
        onnx_opset=onnx_opset,
        convert_qat=convert_qat,
        labels_to_class_mapping=labels_to_class_mapping,
    )


def export(
    model: Module,
    val_loader: DataLoader,
    save_dir: str,
    num_samples: int,
    onnx_opset: int,
    convert_qat: bool,
    labels_to_class_mapping: Optional[Union[str, Dict[int, str]]],
) -> None:
    exporter = ModuleExporter(model, save_dir)

    export_samples = num_samples > 0
    if num_samples < 0:
        num_samples = 1

    for batch, (x, label) in tqdm(
        enumerate(val_loader), desc="Exporting samples", total=num_samples
    ):
        if batch >= num_samples:
            break

        if export_samples:
            exporter.export_samples(
                sample_batches=[x], sample_labels=[label], exp_counter=batch
            )

    _LOGGER.info(f"exporting onnx in {save_dir}")
    exporter.export_onnx(x, opset=onnx_opset, convert_qat=convert_qat)

    exporter.create_deployment_folder(labels_to_class_mapping=labels_to_class_mapping)


if __name__ == "__main__":
    main()
