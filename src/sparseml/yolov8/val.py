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
from sparseml.yolov8.trainers import SparseYOLO
from sparseml.yolov8.utils import data_from_dataset_path


@click.command(
    context_settings=(
        dict(token_normalize_func=lambda x: x.replace("-", "_"), show_default=True)
    )
)
@click.option(
    "--task",
    default="detect",
    type=click.Choice(["detect", "segment", "classify"]),
    help="Specify task to run.",
)
@click.option(
    "--model",
    default="yolov8n.yaml",
    type=str,
    help="i.e. yolov8n.pt, yolov8n.yaml. Path to model file",
)
@click.option(
    "--data",
    default="coco128.yaml",
    type=str,
    help="i.e. coco128.yaml. Path to data file",
)
@click.option(
    "--save-json", default=False, is_flag=True, help="save results to JSON file"
)
@click.option(
    "--save-hybrid",
    default=False,
    is_flag=True,
    help="save hybrid version of labels (labels + additional predictions)",
)
@click.option(
    "--conf",
    default=0.001,
    type=float,
    help="object confidence threshold for detection (default 0.25 predict, 0.001 val)",
)
@click.option(
    "--iou",
    default=0.7,
    type=float,
    help="intersection over union (IoU) threshold for NMS",
)
@click.option(
    "--max_det", default=300, type=int, help="maximum number of detections per image"
)
@click.option("--half", default=False, is_flag=True, help="use half precision (FP16)")
@click.option(
    "--dnn", default=False, is_flag=True, help="use OpenCV DNN for ONNX inference"
)
@click.option("--plots", default=False, is_flag=True, help="show plots during training")
@click.option(
    "--dataset-path",
    type=str,
    default=None,
    help="Path to override default datasets path.",
)
@click.option("--batch", default=16, type=int, help="number of images per batch")
def main(**kwargs):
    if kwargs["dataset_path"] is not None:
        kwargs["data"] = data_from_dataset_path(kwargs["data"], kwargs["dataset_path"])
    del kwargs["dataset_path"]

    model = SparseYOLO(kwargs["model"])
    if hasattr(model, "overrides"):
        model.model.args = model.overrides
    model.val(**kwargs)


if __name__ == "__main__":
    main()
