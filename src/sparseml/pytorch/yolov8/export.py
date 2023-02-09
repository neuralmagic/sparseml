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

import click
from sparseml.pytorch.yolov8.trainers import SparseYOLO


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
    "the models' default input sized will be inferred.",
)
def main(**kwargs):
    model = SparseYOLO(kwargs["model"])
    model.export(**kwargs)


if __name__ == "__main__":
    main()
