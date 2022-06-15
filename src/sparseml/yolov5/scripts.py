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


from yolov5.export import export_run
from yolov5.export import parse_opt as parse_export_args
from yolov5.train import parse_opt as parse_train_args
from yolov5.train import run as train_run
from yolov5.val import parse_opt as parse_val_args
from yolov5.val import val_run


val_onnx_error = None
try:
    from yolov5.val_onnx import parse_opt as parse_val_onnx_args
    from yolov5.val_onnx import val_onnx_run
except Exception as deepsparse_error:
    val_onnx_error = deepsparse_error

__all__ = [
    "train",
    "val",
    "export",
    "val_onnx",
]


def train():
    """
    Hook to call into train.py in YOLOv5 fork
    """
    opt = parse_train_args()
    train_run(**vars(opt))


def val():
    """
    Hook to call into val.py in YOLOv5 fork
    """
    opt = parse_val_args()
    val_run(**vars(opt))


def export():
    """
    Hook to call into export.py in YOLOv5 fork
    """
    opt = parse_export_args()
    export_run(**vars(opt))


def val_onnx():
    """
    Hook to call into val_onnx.py in YOLOv5 fork
    """
    if val_onnx_error:
        raise RuntimeError(val_onnx_error)

    opt = parse_val_onnx_args()
    val_onnx_run(**vars(opt))
