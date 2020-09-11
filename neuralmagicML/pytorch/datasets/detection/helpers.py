"""
Helper classes and functions for PyTorch detection data loaders
"""


import random
import torch

from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as torch_functional
from typing import List, Callable, Tuple, Any

from neuralmagicML.pytorch.utils import ssd_random_crop


__all__ = [
    "AnnotatedImageTransforms",
    "ssd_random_crop_image_and_annotations",
    "random_horizontal_flip_image_and_annotations",
    "detection_collate_fn",
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
    Wraps neuralmagicML.pytorch.utils.ssd_random_crop to work in the
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
    Perorms a horizontal flip on given image and bounding boxes with probability p.
    :param image: the image to randomly flip
    :param annotations: a tuple of bounding boxes and their labels for this image
    :param p: the probability to flip with. Default is 0.5
    :return: A tuple of the randomly flipped image and annotations
    """
    boxes, labels = annotations
    if random.random() < p:
        if labels.numel() > 0:
            boxes[:, [0, 2]] = 1.0 - boxes[:, [2, 0]]  # flip width dimensions
        image = torch_functional.hflip(image)
    return image, (boxes, labels)


def detection_collate_fn(
    batch: List[Any],
) -> Tuple[Tensor, Tuple[Tensor, Tensor, List[Tuple[Tensor, Tensor]]]]:
    """
    Collate function to be used for creating a DataLoader with values transformed by
    encode_annotation_bounding_boxes.
    :param batch: a batch of data points transformed by encode_annotation_bounding_boxes
    :return: the batch stacked as tensors for all values except for the original annotations
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
