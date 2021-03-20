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
Utilities for Yolo V3 pre and post processing for DeepSparse pipelines

Postprocessing is currently tied to yolov3-spp, modify anchor and output
variables if using a different model.
"""


from typing import List, Tuple, Union

import numpy
import torch

import cv2

# ultralytics/yolov5 imports
from utils.general import non_max_suppression


__all__ = [
    "load_image",
    "pre_nms_postprocess",
    "postprocess_nms",
]


def _get_grid(size: int) -> torch.Tensor:
    # adapted from yolov5.yolo.Detect._make_grid
    coords_y, coords_x = torch.meshgrid([torch.arange(size), torch.arange(size)])
    grid = torch.stack((coords_x, coords_y), 2)
    return grid.view(1, 1, size, size, 2).float()


# Yolo V3 specific variables
_YOLO_V3_ANCHORS = [
    torch.Tensor([[10, 13], [16, 30], [33, 23]]),
    torch.Tensor([[30, 61], [62, 45], [59, 119]]),
    torch.Tensor([[116, 90], [156, 198], [373, 326]]),
]
_YOLO_V3_ANCHOR_GRIDS = [t.clone().view(1, -1, 1, 1, 2) for t in _YOLO_V3_ANCHORS]
_YOLO_V3_OUTPUT_SHAPES = [80, 40, 20]
_YOLO_V3_GRIDS = [_get_grid(grid_size) for grid_size in _YOLO_V3_OUTPUT_SHAPES]


def load_image(
    img: Union[str, numpy.ndarray], image_size: Tuple[int] = (640, 640)
) -> numpy.ndarray:
    """
    :param img: file path to image or raw image array
    :param image_size: target shape for image
    :return: Image loaded into numpy and reshaped to the given shape
    """
    img = cv2.imread(img) if isinstance(img, str) else img
    img = cv2.resize(img, image_size)
    img = img[:, :, ::-1].transpose(2, 0, 1)

    return img


def pre_nms_postprocess(outputs: List[numpy.ndarray]) -> torch.Tensor:
    """
    :param outputs: raw outputs of a YOLOv3 model before anchor grid processing
    :return: post-processed model outputs without NMS.
    """
    # postprocess and transform raw outputs into single torch tensor
    processed_outputs = []
    for idx, pred in enumerate(outputs):
        pred = torch.from_numpy(pred)
        pred = pred.sigmoid()

        # get grid and stride
        grid = _YOLO_V3_GRIDS[idx]
        anchor_grid = _YOLO_V3_ANCHOR_GRIDS[idx]
        stride = 640 / _YOLO_V3_OUTPUT_SHAPES[idx]

        # decode xywh box values
        pred[..., 0:2] = (pred[..., 0:2] * 2.0 - 0.5 + grid) * stride
        pred[..., 2:4] = (pred[..., 2:4] * 2) ** 2 * anchor_grid
        # flatten anchor and grid dimensions -> (bs, num_predictions, num_classes + 5)
        processed_outputs.append(pred.view(pred.size(0), -1, pred.size(-1)))
    return torch.cat(processed_outputs, 1)


def postprocess_nms(outputs: torch.Tensor) -> List[numpy.ndarray]:
    """
    :param outputs: Tensor of post-processed model outputs
    :return: List of numpy arrays of NMS predictions for each image in the batch
    """
    # run nms in PyTorch, only post-process first output
    nms_outputs = non_max_suppression(outputs)
    return [output.cpu().numpy() for output in nms_outputs]
