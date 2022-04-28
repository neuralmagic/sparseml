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

from pydantic import BaseModel

from yolov5.utils.general import ROOT


class Yolov5TrainArgs(BaseModel):
    weights: Union[str, Path] = ROOT
    cfg: Union[str, Path] = ""
    data: Union[str, Path] = ROOT
    hyp: Union[str, Path] = ROOT
    epochs: int = 300
    batch_size: int = 16
    imgsz: int = 640
    rect: bool = False
    resume: bool = False
    nosave: bool = False
    noval: bool = False
    noautoanchor: bool = False
    evolve: Tuple[bool, int] = [False, 300]
    bucket: str = ""
    cache: Literal["ram", "disk"] = "ram"
    image_weights: bool = False
    device: str = ""
    multi_scale: bool = False
    single_cls: bool = False
    optimizer: Literal["SGD", "Adam", "AdamW"] = "SGD"
    sync_bin: False
    workers: int = 8
    project: Union[str, Path] = ROOT / "runs/train"
    name: str = "exp"
    exist_ok: False
    quad: False
    cost_lr: False
    label_smoothing: float = 0.0
    patience: int = -1
    freeze: List[int] = [0]
    save_period: int = -1
    local_rank: int = -1
    recipe: Union[str, Path, None] = None
    disable_ema: False
