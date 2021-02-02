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
Helper classes and functions for PyTorch detection data loaders
"""


import random
from typing import Any, Callable, List, Tuple

import torch
from PIL import Image
from torch import Tensor


try:
    from torchvision.transforms import functional as torchvision_functional

    torchvision_import_error = None
except Exception as torchvision_error:
    torchvision_functional = None
    torchvision_import_error = torchvision_error

from sparseml.pytorch.utils import ssd_random_crop


__all__ = [
    "AnnotatedImageTransforms",
    "ssd_random_crop_image_and_annotations",
    "random_horizontal_flip_image_and_annotations",
    "yolo_collate_fn",
    "ssd_collate_fn",
    "bounding_box_and_labels_to_yolo_fmt",
]


class AnnotatedImageTransforms(object):
    """
    Class for chaining transforms that take two parameters
    (images and annotations for object detection).

    :param transforms: List of transformations that take an image and annotation as
        their parameters.
    """

    def __init__(self, transforms: List):
        self._transforms = transforms

    @property
    def transforms(self) -> List[Callable]:
        """
        :return: a list of the transforms performed by this object
        """
        return self._transforms

    def __call__(self, image, annotations):
        for transform in self._transforms:
            image, annotations = transform(image, annotations)
        return image, annotations


def ssd_random_crop_image_and_annotations(
    image: Image.Image, annotations: Tuple[Tensor, Tensor]
) -> Tuple[Image.Image, Tuple[Tensor, Tensor]]:
    """
    Wraps sparseml.pytorch.utils.ssd_random_crop to work in the
    AnnotatedImageTransforms pipeline.

    :param image: the image to crop
    :param annotations: a tuple of bounding boxes and their labels for this image
    :return: A tuple of the cropped image and annotations
    """
    boxes, labels = annotations
    if labels.numel() > 0:
        image, boxes, labels = ssd_random_crop(image, boxes, labels)
    return image, (boxes, labels)


def random_horizontal_flip_image_and_annotations(
    image: Image.Image, annotations: Tuple[Tensor, Tensor], p: float = 0.5
) -> Tuple[Image.Image, Tuple[Tensor, Tensor]]:
    """
    Performs a horizontal flip on given image and bounding boxes with probability p.

    :param image: the image to randomly flip
    :param annotations: a tuple of bounding boxes and their labels for this image
    :param p: the probability to flip with. Default is 0.5
    :return: A tuple of the randomly flipped image and annotations
    """
    if torchvision_import_error is not None:
        raise torchvision_import_error

    boxes, labels = annotations
    if random.random() < p:
        if labels.numel() > 0:
            boxes[:, [0, 2]] = 1.0 - boxes[:, [2, 0]]  # flip width dimensions
        image = torchvision_functional.hflip(image)
    return image, (boxes, labels)


def yolo_collate_fn(
    batch: List[Any],
) -> Tuple[Tensor, Tuple[Tensor, Tensor, List[Tuple[Tensor, Tensor]]]]:
    """
    Collate function to be used for creating a DataLoader with values for Yolo model
    input.

    :param batch: a batch of data points and annotations transformed by
        bounding_box_and_labels_to_yolo_fmt
    :return: the batch stacked as tensors for all values except for the
        original annotations
    """
    images = []
    targets = []
    annotations = []
    for idx, (image, (target, annotation)) in enumerate(batch):
        images.append(image.unsqueeze(0))
        img_label = torch.ones(target.size(0), 1) * idx
        targets.append(torch.cat((img_label, target), 1))
        annotations.append(annotation)

    images = torch.cat(images, 0)
    targets = torch.cat(targets, 0)

    return images, (targets, annotations)


def ssd_collate_fn(
    batch: List[Any],
) -> Tuple[Tensor, Tuple[Tensor, Tensor, List[Tuple[Tensor, Tensor]]]]:
    """
    Collate function to be used for creating a DataLoader with values transformed by
    encode_annotation_bounding_boxes.

    :param batch: a batch of data points transformed by encode_annotation_bounding_boxes
    :return: the batch stacked as tensors for all values except for the
        original annotations
    """
    images = []
    enc_boxes = []
    enc_labels = []
    annotations = []

    for image, (enc_box, enc_label, annotation) in batch:
        images.append(image.unsqueeze(0))
        enc_boxes.append(enc_box.unsqueeze(0))
        enc_labels.append(enc_label.unsqueeze(0))
        annotations.append(annotation)

    images = torch.cat(images, 0)
    enc_boxes = torch.cat(enc_boxes, 0)
    enc_labels = torch.cat(enc_labels, 0)

    return images, (enc_boxes, enc_labels, annotations)


def bounding_box_and_labels_to_yolo_fmt(annotations):
    boxes, labels = annotations

    if boxes.numel() == 0:
        return torch.zeros(0, 5)

    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    return torch.stack((labels, cx, cy, w, h)).T
