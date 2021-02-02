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
Helper functions and classes for creating and training PyTorch Yolo models
"""


from typing import Iterable, List, Tuple

import torch
from torch import Tensor


try:
    from torchvision.ops.boxes import batched_nms
except Exception:
    batched_nms = None


all = [
    "get_output_grid_shapes",
    "yolo_v3_anchor_groups",
    "build_targets",
    "box_giou",
    "YoloGrids",
    "postprocess_yolo",
]


def get_output_grid_shapes(outputs: List[Tensor]) -> List[Tensor]:
    """
    :param outputs: List of Yolo model outputs
    :return: A list of the grid dimensions for each of the Yolo outputs
    """
    return [Tensor(list(output.shape[2:4])) for output in outputs]


def yolo_v3_anchor_groups() -> List[Tensor]:
    """
    :return: List of the default anchor coordinate groups for Yolo V3 outputs
    """
    return [
        Tensor([[116, 90], [156, 198], [373, 326]]),
        Tensor([[30, 61], [62, 45], [59, 119]]),
        Tensor([[10, 13], [16, 30], [33, 23]]),
    ]


def _width_height_iou(wh_a: Tensor, wh_b: Tensor) -> Tensor:
    # [n,2], [m,2] -> [n,m]
    wh_a = wh_a.unsqueeze(1)
    wh_b = wh_b.unsqueeze(0)

    area_a = wh_a.prod(2)
    area_b = wh_b.prod(2)

    intersection = torch.min(wh_a, wh_b).prod(2)
    return intersection / (area_a + area_b - intersection)


def build_targets(
    targets: Tensor,
    anchors_groups: List[Tensor],
    grid_shapes: List[Tensor],
    iou_threshold: float = 0.2,
) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
    """
    Returns a representation of the image targets according to the given
    anchor groups and grid shapes.

    :param targets: Yolo data targets tensor of shape n,6 with columns image number,
        class, center_x, center_y, width, height
    :param anchors_groups: List of n,2 Tensors of anchor point coordinates for
        each of the Yolo model's detectors
    :param grid_shapes: List of n,2 Tensors of the Yolo models output grid shapes
        for a particular input shape
    :param iou_threshold: the minimum IoU value to consider an object box to match
        to an anchor point. Default is 0.2
    :return:
    """
    num_targets = targets.shape[0]
    num_anchors = len(anchors_groups[0])
    classes, boxes, indices, target_anchors = [], [], [], []

    # copy targets for each anchor
    anchor_indices = (
        torch.arange(num_anchors, device=targets.device)
        .float()
        .view(num_anchors, 1)
        .repeat(1, num_targets)
    )
    targets = torch.cat(
        (targets.repeat(num_anchors, 1, 1), anchor_indices[:, :, None]), 2
    )

    offset_bias = 0.5
    offset_values = (
        torch.tensor(
            [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],  # None, j,k,l,m
            device=targets.device,
        ).float()
        * offset_bias
    )  # offsets

    grid_scale = torch.ones(7, device=targets.device)  # tensor for grid space scaling

    for idx, anchors in enumerate(anchors_groups):
        # scale targets to current grid
        anchors = anchors.to(targets.device)
        grid_scale[2:6] = torch.tensor(grid_shapes[idx])[[0, 1, 0, 1]]
        scaled_targets = targets * grid_scale

        if num_targets:
            # mask non-matches
            wh_iou_mask = (
                _width_height_iou(anchors, scaled_targets[0, :, 4:6]) > iou_threshold
            )
            scaled_targets = scaled_targets[wh_iou_mask]

            # adjust for offsets for grid index rounding
            targets_xy = scaled_targets[:, 2:4]
            targets_xy_inv = grid_scale[[2, 3]] - targets_xy
            j, k = ((targets_xy % 1.0 < offset_bias) & (targets_xy > 1.0)).t()
            l, m = ((targets_xy_inv % 1.0 < offset_bias) & (targets_xy_inv > 1.0)).t()
            offset_filter = torch.stack((torch.ones_like(j), j, k, l, m))
            scaled_targets = scaled_targets.repeat((5, 1, 1))[offset_filter]
            offsets = (torch.zeros_like(targets_xy)[None] + offset_values[:, None])[
                offset_filter
            ]
        else:
            scaled_targets = targets[0]
            offsets = 0

        # extract fields
        image, clazz = scaled_targets[:, :2].long().t()
        targets_xy = scaled_targets[:, 2:4]
        targets_wh = scaled_targets[:, 4:6]
        grid_indices = (targets_xy - offsets).long()
        grid_x, grid_y = grid_indices.t()
        anchor_idxs = scaled_targets[:, 6].long()

        indices.append((image, anchor_idxs, grid_x, grid_y))
        boxes.append(torch.cat((targets_xy - grid_indices.float(), targets_wh), 1))
        target_anchors.append(anchors[anchor_idxs])
        classes.append(clazz)

    return classes, boxes, indices, target_anchors


def box_giou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """
    :param boxes_a: 4,N Tensor of xywh bounding boxes
    :param boxes_b: 4,N Tensor of xywh bounding boxes
    :return: Shape N Tensor of GIoU values between boxes in the input tensors
    """
    # get ltrb coordinates
    lt_x_a = boxes_a[0] - boxes_a[2] / 2.0
    lt_y_a = boxes_a[1] - boxes_a[3] / 2.0
    rb_x_a = boxes_a[0] + boxes_a[2] / 2.0
    rb_y_a = boxes_a[1] + boxes_a[3] / 2.0
    lt_x_b = boxes_b[0] - boxes_b[2] / 2.0
    lt_y_b = boxes_b[1] - boxes_b[3] / 2.0
    rb_x_b = boxes_b[0] + boxes_b[2] / 2.0
    rb_y_b = boxes_b[1] + boxes_b[3] / 2.0

    # base IoU
    inter = (torch.min(rb_x_a, rb_x_b) - torch.max(lt_x_a, lt_x_b)).clamp(0) * (
        torch.min(rb_y_a, rb_y_b) - torch.max(lt_y_a, lt_y_b)
    ).clamp(0)
    area_a = boxes_a[2] * boxes_a[3]
    area_b = boxes_b[2] * boxes_b[3]
    union = area_a + area_b - inter + 1e-9
    iou = inter / union

    # convex area
    convex_w = torch.max(rb_x_a, rb_x_b) - torch.min(lt_x_a, lt_x_b)
    convex_h = torch.max(rb_y_a, rb_y_b) - torch.min(lt_y_a, lt_y_b)
    convex_area = convex_w * convex_h + 1e-9  # convex area

    return iou - (convex_area - union) / convex_area  # GIoU


class YoloGrids(object):
    """
    Helper class to compute and store Yolo output and anchor box grids

    :param anchor_groups: List of n,2 tensors of the Yolo model's anchor points
        for each output group. Defaults to yolo_v3_anchor_groups
    """

    def __init__(self, anchor_groups: List[Tensor] = None):
        self._grids = {}
        anchor_groups = anchor_groups or yolo_v3_anchor_groups()
        self._anchor_grids = [
            t.clone().view(1, -1, 1, 1, 2) for t in yolo_v3_anchor_groups()
        ]

    def get_grid(self, size_x: int, size_y: int) -> Tensor:
        """
        :param size_x: grid size x
        :param size_y: grid size y
        :return: Yolo output box grid for size x,y to be used for model output decoding.
            will have shape (1, 1, size_y, size_x, 2)
        """
        grid_shape = (size_x, size_y)
        if grid_shape not in self._grids:
            coords_y, coords_x = torch.meshgrid(
                [torch.arange(size_y), torch.arange(size_x)]
            )
            grid = torch.stack((coords_x, coords_y), 2)
            self._grids[grid_shape] = grid.view(1, 1, size_y, size_x, 2)

        return self._grids[grid_shape]

    def get_anchor_grid(self, group_idx: int) -> Tensor:
        """
        :param group_idx: Index of output group for this anchor grid
        :return: grid tensor of shape 1, num_anchors, 1, 1, 2
        """
        return self._anchor_grids[group_idx]

    def num_anchor_grids(self) -> int:
        """
        :return: The number of anchor grids available (number of yolo model outputs)
        """
        return len(self._anchor_grids)


def _xywh_to_ltrb(boxes, in_place: bool = False):
    if not in_place:
        boxes = boxes.clone()
    boxes[:, 0], boxes[:, 2] = (  # ltrb x
        boxes[:, 0] - boxes[:, 2] / 2.0,
        boxes[:, 0] + boxes[:, 2] / 2.0,
    )
    boxes[:, 1], boxes[:, 3] = (  # ltrb y
        boxes[:, 1] - boxes[:, 3] / 2.0,
        boxes[:, 1] + boxes[:, 3] / 2.0,
    )
    return boxes


def postprocess_yolo(
    preds: List[Tensor],
    input_shape: Iterable[int],
    yolo_grids: YoloGrids = None,
    confidence_threshold: float = 0.1,
    iou_threshold: float = 0.6,
    max_detections: int = 300,
) -> List[Tuple[Tensor, Tensor, Tensor]]:
    """
    Decode the outputs of a Yolo model and perform non maximum suppression
    on the predicted boxes.

    :param preds: list of Yolo model output tensors
    :param input_shape: shape of input image to model. Default is [640, 640]
    :param yolo_grids: optional YoloGrids object for caching previously used grid shapes
    :param confidence_threshold: minimum confidence score for a prediction to be
        considered a detection. Default is 0.1
    :param iou_threshold: IoU threshold for non maximum suppression. Default is 0.6
    :param max_detections: maximum number of detections after nms. Default is 300
    :return: List of predicted bounding boxes (n,4), labels, and scores for each output
        in the batch
    """
    if batched_nms is None:
        raise RuntimeError(
            "Unable to import batched_nms from torchvision.ops try upgrading your"
            " torch and torchvision versions"
        )
    yolo_grids = yolo_grids or YoloGrids()

    # decode each of the model output grids then concatenate
    outputs = []
    for idx, pred in enumerate(preds):
        pred = pred.sigmoid()

        # build grid and calculate stride
        grid_shape = pred.shape[2:4]
        grid = yolo_grids.get_grid(*grid_shape)
        anchor_grid = yolo_grids.get_anchor_grid(idx)
        stride = input_shape[0] / grid_shape[0]

        # decode xywh box values
        pred[..., 0:2] = (pred[..., 0:2] * 2.0 - 0.5 + grid) * stride
        pred[..., 2:4] = (pred[..., 2:4] * 2) ** 2 * anchor_grid
        # flatten anchor and grid dimensions -> (bs, num_predictions, num_classes + 5)
        outputs.append(pred.view(pred.size(0), -1, pred.size(-1)))
    outputs = torch.cat(outputs, 1)

    # perform nms on each image in batch
    nms_outputs = []
    for image_idx, output in enumerate(outputs):
        # filter out low confidence predictions
        confidence_mask = output[..., 4] > confidence_threshold
        output = output[confidence_mask]

        if output.size(0) == 0:  # no predictions, return empty tensor
            nms_outputs.append(torch.empty(0, 6))
            continue

        # scale class confidences by object confidence, convert to ltrb
        output[:, 5:] *= output[:, 4:5]
        _xywh_to_ltrb(output[:, :4], in_place=True)

        # attach labels of all positive predictions
        class_confidence_mask = output[:, 5:] > confidence_threshold
        pred_idxs, class_idxs = class_confidence_mask.nonzero(as_tuple=False).t()
        output = torch.cat(
            [
                output[pred_idxs, :4],
                output[pred_idxs, class_idxs + 5].unsqueeze(-1),
                class_idxs.float().unsqueeze(-1),
            ],
            1,
        )

        if output.size(0) == 0:  # no predictions, return empty tensor
            nms_outputs.append(torch.empty(0, 6))
            continue

        # run nms
        nms_filter = batched_nms(  # boxes, scores, labels, threshold
            output[:, :4], output[:, 4], output[:, 5], iou_threshold
        )
        if nms_filter.size(0) > max_detections:
            nms_filter = nms_filter[:max_detections]
        output = output[nms_filter]

        # extract outputs, rescale boxes to [0, 1]
        boxes = output[:, :4]
        boxes[:, [0, 2]] /= input_shape[0]  # scale x
        boxes[:, [1, 3]] /= input_shape[1]  # scale y
        labels = output[:, 5].long()
        scores = output[:, 4]

        nms_outputs.append((boxes, labels, scores))

    return nms_outputs
