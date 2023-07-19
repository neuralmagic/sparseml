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

import logging

import click
from sparseml.yolov8.trainers import SparseYOLO
from sparseml.yolov8.utils import data_from_dataset_path


logger = logging.getLogger()

# Options generated from
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/configs/default.yaml


@click.command(
    context_settings=(
        dict(token_normalize_func=lambda x: x.replace("-", "_"), show_default=True)
    )
)
@click.option("--recipe", default=None, type=str, help="Path to recipe")
@click.option(
    "--recipe-args",
    default=None,
    type=str,
    help="json parsable dict of recipe variable names to values to overwrite with",
)
@click.option(
    "--resume", default=False, is_flag=True, help="resume training from last checkpoint"
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
    help="i.e. yolov8n.pt, yolov8n.yaml, yolov8n-seg.yaml. Path to model file",
)
@click.option(
    "--data",
    default="coco128.yaml",
    type=str,
    help="i.e. coco128.yaml, coco128-seg.yaml. Path to data file",
)
@click.option("--epochs", default=100, type=int, help="number of epochs to train for")
@click.option(
    "--patience",
    default=50,
    type=int,
    help="epochs to wait for no observable improvement for early stopping of training",
)
@click.option("--batch", default=16, type=int, help="number of images per batch")
@click.option("--imgsz", default=640, type=int, help="size of input images")
@click.option("--save", default=True, is_flag=True, help="save checkpoints")
@click.option("--cache", default=False, is_flag=True, help="Use cache for data loading")
@click.option(
    "--device",
    default=None,
    type=str,
    help="cuda device, i.e. 0 or 0,1,2,3 or cpu. Device to run on",
)
@click.option(
    "--workers", default=8, type=int, help="number of worker threads for data loading"
)
@click.option("--project", default=None, type=str, help="project name")
@click.option("--name", default=None, type=str, help="experiment name")
@click.option(
    "--exist-ok",
    default=False,
    is_flag=True,
    help="whether to overwrite existing experiment",
)
@click.option(
    "--pretrained",
    default=False,
    is_flag=True,
    help="whether to use a pretrained model",
)
@click.option(
    "--optimizer",
    default="SGD",
    type=click.Choice(["SGD", "Adam", "AdamW", "RMSProp"]),
    help="optimizer to use",
)
@click.option(
    "--verbose", default=False, is_flag=True, help="whether to print verbose output"
)
@click.option("--seed", default=0, type=int, help="random seed for reproducibility")
@click.option(
    "--deterministic",
    default=True,
    is_flag=True,
    help="whether to enable deterministic mode",
)
@click.option(
    "--single-cls",
    default=False,
    is_flag=True,
    help="train multi-class data as single-class",
)
@click.option(
    "--image-weights",
    default=False,
    is_flag=True,
    help="use weighted image selection for training",
)
@click.option(
    "--rect", default=False, is_flag=True, help="support rectangular training"
)
@click.option(
    "--cos-lr", default=False, is_flag=True, help="use cosine learning rate scheduler"
)
@click.option(
    "--close-mosaic",
    default=10,
    type=int,
    help="disable mosaic augmentation for final 10 epochs",
)
@click.option(
    "--overlap-mask",
    default=True,
    is_flag=True,
    help="masks should overlap during training",
)
@click.option("--mask-ratio", default=4, type=int, help="mask downsample ratio")
@click.option("--dropout", default=0.0, type=float, help="use dropout regularization")
@click.option(
    "--lr0",
    default=0.01,
    type=float,
    help="initial learning rate (SGD=1E-2, Adam=1E-3)",
)
@click.option(
    "--lrf", default=0.01, type=float, help="final OneCycleLR learning rate (lr0 * lrf)"
)
@click.option("--momentum", type=float, default=0.937, help="SGD momentum/Adam beta1")
@click.option(
    "--weight-decay", type=float, default=0.0005, help="optimizer weight decay 5e-4"
)
@click.option(
    "--warmup-epochs", type=float, default=3.0, help="warmup epochs (fractions ok)"
)
@click.option(
    "--warmup-momentum", type=float, default=0.8, help="warmup initial momentum"
)
@click.option(
    "--warmup-bias-lr", type=float, default=0.1, help="warmup initial bias lr"
)
@click.option("--box", type=float, default=7.5, help="box loss gain")
@click.option(
    "--cls", type=float, default=0.5, help="cls loss gain (scale with pixels)"
)
@click.option("--dfl", type=float, default=1.5, help="dfl loss gain")
@click.option(
    "--fl-gamma",
    type=float,
    default=0.0,
    help="focal loss gamma (efficientDet default gamma=1.5)",
)
@click.option("--label-smoothing", type=float, default=0.0)
@click.option("--nbs", type=float, default=64, help="nominal batch size")
@click.option(
    "--hsv-h", type=float, default=0.015, help="image HSV-Hue augmentation (fraction)"
)
@click.option(
    "--hsv-s",
    type=float,
    default=0.7,
    help="image HSV-Saturation augmentation (fraction)",
)
@click.option(
    "--hsv-v", type=float, default=0.4, help="image HSV-Value augmentation (fraction)"
)
@click.option("--degrees", type=float, default=0.0, help="image rotation (+/- deg)")
@click.option(
    "--translate", type=float, default=0.1, help="image translation (+/- fraction)"
)
@click.option("--scale", type=float, default=0.5, help="image scale (+/- gain)")
@click.option("--shear", type=float, default=0.0, help="image shear (+/- deg)")
@click.option(
    "--perspective",
    type=float,
    default=0.0,
    help="image perspective (+/- fraction), range 0-0.001",
)
@click.option(
    "--flipud", type=float, default=0.0, help="image flip up-down (probability)"
)
@click.option(
    "--fliplr", type=float, default=0.5, help="image flip left-right (probability)"
)
@click.option("--mosaic", type=float, default=1.0, help="image mosaic (probability)")
@click.option("--mixup", type=float, default=0.0, help="image mixup (probability)")
@click.option(
    "--copy-paste", type=float, default=0.0, help="segment copy-paste (probability)"
)
@click.option(
    "--dataset-path",
    type=str,
    default=None,
    help="Path to override default dataset path.",
)
def main(**kwargs):
    if kwargs["dataset_path"] is not None:
        kwargs["data"] = data_from_dataset_path(kwargs["data"], kwargs["dataset_path"])
    del kwargs["dataset_path"]

    # NOTE: the task, model, and data kwargs will override the values in default.yaml
    # They should be provided or default values will be used.
    model = SparseYOLO(kwargs["model"], kwargs["task"])
    model.train(**kwargs)


if __name__ == "__main__":
    main()
