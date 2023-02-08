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

from typing import Any, Dict, Iterable

import numpy
from torch import Tensor

from sparseml.utils.datasets import (
    CIFAR_10_CLASSES,
    COCO_CLASSES,
    COCO_CLASSES_80,
    IMAGENET_CLASSES,
    IMAGENETTE_CLASSES,
    VOC_CLASSES,
)


__all__ = [
    "apply_one_hot_label_mapping",
    "cifar10_label_mapping",
    "imagenette_label_mapping",
    "imagenet_label_mapping",
    "mnist_label_mapping",
    "coco_yolo_2017_mapping",
    "coco_mapping",
]

##############################
#
# Callbacks for mapping labels
#
##############################


def apply_one_hot_label_mapping(labels: Tensor, class_names: Dict[Any, str]):
    def _apply_label(label: int):
        one_hot_label = [0] * len(class_names.keys())
        one_hot_label[label] = 1
        return one_hot_label

    arr = [
        numpy.array([_apply_label(label) for label in labels]),
        numpy.array([[val for _, val in class_names.items()]] * len(labels)),
    ]

    return arr


def apply_box_label_mapping(labels: Iterable[Tensor], class_names: Dict[Any, str]):
    class_names = [
        class_names[i] if i in class_names else ""
        for i in range(max(class_names.keys()) + 1)
    ]
    return [
        labels[0],
        labels[1],
        [numpy.array([class_names] * labels[0].shape[0])],
    ]


def cifar10_label_mapping(labels: Tensor):
    return apply_one_hot_label_mapping(labels, CIFAR_10_CLASSES)


def imagenette_label_mapping(labels: Tensor):
    return apply_one_hot_label_mapping(
        labels,
        IMAGENETTE_CLASSES,
    )


def imagenet_label_mapping(labels: Tensor):
    return apply_one_hot_label_mapping(
        labels,
        IMAGENET_CLASSES,
    )


def mnist_label_mapping(labels: Tensor):
    return apply_one_hot_label_mapping(labels, {idx: str(idx) for idx in range(10)})


def coco_yolo_2017_mapping(labels: Iterable[Tensor]):
    class_names = [val for _, val in COCO_CLASSES_80.items()]

    return [
        labels[0],
        [numpy.array([class_names] * labels[0].shape[0])],
    ]


def coco_mapping(labels: Iterable[Tensor]):
    return apply_box_label_mapping(labels, COCO_CLASSES)


def voc_mapping(labels: Iterable[Tensor]):
    return apply_box_label_mapping(labels, VOC_CLASSES)
