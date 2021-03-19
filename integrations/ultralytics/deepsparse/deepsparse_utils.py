"""
Utilities for Yolo V3 pre and post processing for DeepSparse pipelines

Postprocessing is currently tied to yolov3-spp, modify anchor and output
variables if using a different model.
"""


import cv2
import numpy
import torch
from typing import List, Tuple

# ultralytics/yolov5 imports
from utils.general import non_max_suppression


__all__ = [
    "preprocess_images",
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


def _preprocess_image(
    img: numpy.ndarray,
    image_size: Tuple[int] = (640, 640),
    fp32: bool = True,
) -> numpy.ndarray:
    # raw numpy image from cv2.imread -> preprocessed floats w/ shape (3, (image_size))
    img = cv2.resize(img, image_size)
    img = img[:, :, ::-1].transpose(2, 0, 1)

    return img if not fp32 else img.astype(numpy.float32) / 255.0


def preprocess_images(
    images: List[numpy.ndarray],
    image_size: Tuple[int] = (640, 640),
    fp32: bool = True,
) -> numpy.ndarray:
    """
    :param images: List of raw numpy uint8 images loaded from cv2.imread
    :param image_size: Size images should be resized to
    :param fp32: set True to cast to float32 and divide by 255.0
    :return: array of shape (n_images, 3, *image_size) that is the preprocessed
        batch of images
    """
    # list of raw numpy images -> preprocessed batch of images
    images = [_preprocess_image(img, image_size, fp32) for img in images]

    batch = numpy.stack(images)  # shape: (batch_size, 3, (image_size))
    return numpy.ascontiguousarray(batch)


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
    return [output.numpy() for output in nms_outputs]