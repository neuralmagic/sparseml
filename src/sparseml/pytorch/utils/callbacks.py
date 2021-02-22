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

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from sparseml.utils.class_mappings import (
    CIFAR_10_CLASSES,
    COCO_CLASSES,
    COCO_CLASSES_80,
    IMAGENET_CLASSES,
    IMAGENETTE_CLASSES,
    VOC_CLASSES,
)


__all__ = [
    "iter_dataset_with_orig_wrapper",
    "iterable_data_split_cb",
    "IterableDatasetWrapper",
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
# Callbacks for wrapping datasets
#
##############################
def iter_dataset_with_orig_wrapper(
    get_original_cb: Optional[Callable] = None,
) -> Callable[[Dataset], Dataset]:
    """
    Creates a wrapper function which, when passed an iterable dataset, returns a
        wrapped dataset. The wrapped dataset returns untransformed input data at
        the end of the array.
    :param get_original_cb: Callback function for returning the original input
        data from the dataset. When set to None, uses `disable_transform_cb`
        function.
    :return: The wrapper function
    """
    if get_original_cb is None:
        get_original_cb = disable_transform_cb()

    def _iter_dataset_with_orig_wrapper(dataset: Dataset):
        return IterableDatasetWrapper(dataset, get_original_cb=get_original_cb)

    return _iter_dataset_with_orig_wrapper


class IterableDatasetWrapper(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        get_original_cb: Callable,
    ):
        self.dataset = dataset
        self.get_original_cb = get_original_cb

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if self.get_original_cb:
            original_sample = self.get_original_cb(self.dataset, idx)
        else:
            original_sample = None

        return data + (original_sample,)


def disable_transform_cb(
    max_width=600, max_height=600
) -> Callable[[Dataset, int], numpy.ndarray]:
    """
    Creates a callable function which, given an iterable image dataset and
        an index idx, returns the untransformed but padded image at that index.
        Assumes that the transformation function is attribute 'transform'
        in dataset and that the image is index 0 in the getter.
    :param max_width: max width of padded image. If image width is higher it
        will be resized to fit max dimension.
    :param max_height: max height of padded image. If image height is higher
        it will be resized to fit max dimension.
    :return: callable function
    """

    def _disable_transform(dataset: Dataset, idx: int) -> numpy.ndarray:
        if hasattr(dataset, "transform") or hasattr(dataset, "transforms"):
            transform_cache = getattr(dataset, "transform")  # dataset.transform
            transforms_cache = getattr(dataset, "transforms")
            # dataset.transform = None
            setattr(dataset, "transform", None)
            setattr(dataset, "transforms", None)
            original_sample = dataset[idx][0]

            # Pads original image
            width, height = original_sample.size
            original_sample = original_sample.resize(
                (min(width, max_width), min(max_height, height))
            )
            padding_left = int((max_width - width) / 2)

            padding_top = int((max_height - height) / 2)
            result = Image.new(original_sample.mode, (max_width, max_height), color=0)
            result.paste(original_sample, (padding_left, padding_top))

            original_sample = numpy.array(result)

            # dataset.transform = transform_cache
            setattr(dataset, "transform", transform_cache)
            setattr(dataset, "transforms", transforms_cache)
        else:
            original_sample = numpy.array(dataset[idx][0])
        return original_sample

    return _disable_transform


##############################
#
# Callbacks for splitting input/label data
#
##############################
def iterable_data_split_cb(
    data: List, label_index: int = 1
) -> Tuple[List[Any], List[Any]]:
    """
    Split a list into a tuple of lists where:
        - first list contains the first element in the list
        - second list contains the rest of the elements in the list
    """
    return (tuple(data[:label_index]), data[label_index:])


##############################
#
# Callbacks for mapping labels
#
##############################
def apply_one_hot_label_mapping(labels: List[Tensor], class_names: Dict[Any, str]):
    def _apply_label(label: int):
        one_hot_label = [0] * len(class_names.keys())
        one_hot_label[label] = 1
        return one_hot_label

    arr = [
        numpy.array([_apply_label(label[0]) for label in labels]),
        numpy.array([[val for _, val in class_names.items()]] * len(labels)),
    ]

    return arr


def apply_box_label_mapping(labels: List[List[Tensor]], class_names: Dict[Any, str]):
    class_names = [
        class_names[i] if i in class_names else ""
        for i in range(max(class_names.keys()) + 1)
    ]

    return [
        [label[0] for label in labels],
        [label[1] for label in labels],
        [numpy.array([class_names] * labels[0][0].shape[0])],
    ]


def cifar10_label_mapping(labels: List[Tensor]):
    return apply_one_hot_label_mapping(labels, CIFAR_10_CLASSES)


def imagenette_label_mapping(labels: List[Tensor]):
    return apply_one_hot_label_mapping(
        labels,
        IMAGENETTE_CLASSES,
    )


def imagenet_label_mapping(labels: List[Tensor]):
    return apply_one_hot_label_mapping(
        labels,
        IMAGENET_CLASSES,
    )


def mnist_label_mapping(labels: List[Tensor]):
    return apply_one_hot_label_mapping(labels, {idx: str(idx) for idx in range(10)})


def coco_yolo_2017_mapping(labels: List[List[Tensor]]):
    class_names = [val for _, val in COCO_CLASSES_80.items()]

    return [
        [label[0] for label in labels],
        [numpy.array([class_names] * labels[0][0].shape[0])],
    ]


def coco_mapping(labels: List[List[Tensor]]):
    return apply_box_label_mapping(labels, COCO_CLASSES)


def voc_mapping(labels: List[List[Tensor]]):
    return apply_box_label_mapping(labels, VOC_CLASSES)
