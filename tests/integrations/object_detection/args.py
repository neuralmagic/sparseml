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

from pathlib import Path
from typing import List, Literal, Tuple, Union

from pydantic import BaseModel, Field

from yolov5.utils.general import ROOT


class Yolov5TrainArgs(BaseModel):
    weights: Union[str, Path] = Field(default='""', description="initial weights path")
    cfg: Union[str, Path] = Field(default='""', description="model.yaml path")
    data: Union[str, Path] = Field(
        default=ROOT / "data/coco128.yaml", description="dataset.yaml path"
    )
    hyp: Union[str, Path] = Field(
        default=ROOT / "data/hyps/hyp.scratch-low.yaml",
        description="hyperparameters path",
    )
    epochs: int = Field(default=300)
    batch_size: int = Field(
        default=16, description="total batch size for all GPUs, -1 for autobatch"
    )
    imgsz: int = Field(default=640, description="train, val image size (pixels)")
    rect: bool = Field(default=False, description="rectangular training")
    resume: bool = Field(default=False, description="resume most recent training")
    nosave: bool = Field(default=False, description="only save final checkpoint")
    noval: bool = Field(default=False, description="only validate final epoch")
    noautoanchor: bool = Field(default=False, description="disable AutoAnchor")
    evolve: Tuple[bool, int] = Field(
        default=[False, 300], description="evolve hyperparameters for x generations"
    )
    bucket: str = Field(default='""', description="gsutil bucket")
    cache: Literal["ram", "disk"] = Field(
        default="ram", description='--cache images in "ram" (default) or "disk"'
    )
    image_weights: bool = Field(
        default=False, description="use weighted image selection for training"
    )
    device: str = Field(default="", description="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    multi_scale: bool = Field(default=False, description="vary img-size +/- 50%%")
    single_cls: bool = Field(
        default=False, description="train multi-class data as single-class"
    )
    optimizer: Literal["SGD", "Adam", "AdamW"] = Field(
        default="SGD", description="optimizer"
    )
    sync_bin: bool = Field(
        default=False, description="use SyncBatchNorm, only available in DDP mode"
    )
    workers: int = Field(
        default=8, description="max dataloader workers (per RANK in DDP mode)"
    )
    project: Union[str, Path] = Field(
        default=ROOT / "test_runs/train", description="save to project/name"
    )
    name: str = Field(default="exp", description="save to project/name")
    exist_ok: bool = Field(
        default=False, description="existing project/name ok, do not increment"
    )
    quad: bool = Field(default=False, description="quad dataloader")
    cost_lr: bool = Field(default=False, description="cosine LR scheduler")
    label_smoothing: float = Field(default=0.0, description="Label smoothing epsilon")
    patience: int = Field(
        default=100, description="EarlyStopping patience (epochs without improvement)"
    )
    freeze: str = Field(
        default="0", description="Freeze layers: backbone=10, first3=0 1 2"
    )
    save_period: int = Field(
        default=-1, description="Save checkpoint every x epochs (disabled if < 1)"
    )
    recipe: Union[str, Path, None] = Field(
        default=None,
        description="Path to a sparsification recipe, "
        "see https://github.com/neuralmagic/sparseml for more information",
    )
    upload_dataset: bool = Field(
        default=False, description='W&B: Upload data, "val" option'
    )
    disable_ema: bool = Field(
        default=False, description="Disable EMA model updates (enabled by default)"
    )


class Yolov5ExportArgs(BaseModel):
    weights: Union[str, Path] = Field(
        default=ROOT / "yolov5s.pt", description="initial weights path"
    )
    imgsz: List[int] = Field(default=[640, 640], description="image (h, w)")
    batch_size: int = Field(default=16, description="batch size")
    device: str = Field(
        default="cpu", description="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    half: bool = Field(default=False, description="FP16 half-precision export")
    inplace: bool = Field(default=False, description="set YOLOv5 Detect() inplace=True")
    train: bool = Field(default=False, description="model.train() mode")
    optimize: bool = Field(
        default=False, description="TorchScript: optimize for mobile"
    )
    int8: bool = Field(default=False, description="CoreML/TF INT8 quantization")
    dynamic: bool = Field(default=False, description="ONNX/TF: dynamic axes")
    simplify: bool = Field(default=False, description="ONNX: simplify model")
    opset: int = Field(default=12, description="ONNX: opset version")
    verbose: bool = Field(default=False, description="TensorRT: verbose log")
    nms: bool = Field(default=False, description="TF: add NMS to model")
    agnostic_nms: bool = Field(
        default=False, description="TF: add agnostic NMS to model"
    )
    remove_grid: bool = Field(
        default=False, description="remove export of Detect() layer grid"
    )
