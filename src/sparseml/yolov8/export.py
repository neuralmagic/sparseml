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


import os

import click
from sparseml.yolov8.trainers import SparseYOLO


# Options generated from
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/configs/default.yaml


@click.command(
    context_settings=(
        dict(token_normalize_func=lambda x: x.replace("-", "_"), show_default=True)
    )
)
@click.option("--model", required=True, type=str, help="Path to .pt model file")
@click.option(
    "--save-dir",
    default="exported",
    type=str,
    help="The directory to save the exported ONNX deployment to. "
    "Defaults to `exported`",
)
@click.option(
    "--opset",
    default=13,
    type=int,
    help="The opset version to export the ONNX model with. " "Defaults to 13",
)
@click.option(
    "--imgsz",
    default=None,
    type=int,
    help="Size of input images. If not specified, "
    "the default input sized will be inferred from"
    "the model.",
)
@click.option(
    "--one-shot",
    default=None,
    type=str,
    help="Path to recipe to apply in a zero shot fashion. Defaults to None.",
)
@click.option(
    "--export-samples",
    type=int,
    default=0,
    help="Number of samples to export with onnx",
)
@click.option(
    "--save-one-shot-torch",
    default=False,
    help="If one-shot recipe is supplied and "
    "this flag is set to True,the torch model with "
    "the one-shot recipe applied will be exported.",
)
@click.option(
    "--deterministic",
    default=True,
    is_flag=True,
    help="whether to enable deterministic mode",
)
@click.option(
    "--device",
    default=None,
    type=str,
    help="cuda device, i.e. 0 or 0,1,2,3 or cpu. Device to run on",
)
@click.option(
    "--dataset-path",
    type=str,
    default=None,
    help="Path to override default dataset path.",
)
def main(**kwargs):

    if not os.path.isfile(kwargs["model"]):
        raise ValueError(
            "model must be a path to an individual model file such as `model.pt`, "
            f"{kwargs['model']} either is a directory or does not exist"
        )

    model = SparseYOLO(kwargs["model"])
    model.export(**kwargs)


if __name__ == "__main__":
    main()
