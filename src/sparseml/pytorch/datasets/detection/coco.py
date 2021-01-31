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

import os

import torch


try:
    from torchvision.transforms import ColorJitter
    from torchvision.transforms import functional as torch_functional

    torchvision_import_error = None
except Exception as torchvision_error:
    ColorJitter = None
    torch_functional = None
    torchvision_import_error = torchvision_error

try:
    from torchvision.datasets import CocoDetection

    import pycocotools
except Exception:
    CocoDetection = object
    pycocotools = None

import urllib.request as request
import zipfile

from sparseml.pytorch.datasets.detection.helpers import (
    AnnotatedImageTransforms,
    bounding_box_and_labels_to_yolo_fmt,
    random_horizontal_flip_image_and_annotations,
    ssd_random_crop_image_and_annotations,
)
from sparseml.pytorch.datasets.registry import DatasetRegistry
from sparseml.pytorch.utils import DefaultBoxes, get_default_boxes_300
from sparseml.utils.datasets import (
    IMAGENET_RGB_MEANS,
    IMAGENET_RGB_STDS,
    default_dataset_path,
)


__all__ = [
    "CocoDetectionDataset",
    "coco_2017_yolo",
    "COCO_CLASSES",
    "COCO_CLASSES_80",
]


COCO_IMAGE_ZIP_ROOT = "http://images.cocodataset.org/zips"
COCO_ANNOTATION_ZIP_ROOT = "http://images.cocodataset.org/annotations"


@DatasetRegistry.register(
    key=["coco_detection", "coco"],
    attributes={
        "transform_means": IMAGENET_RGB_MEANS,
        "transform_stds": IMAGENET_RGB_STDS,
        "num_classes": 91,
    },
)
class CocoDetectionDataset(CocoDetection):
    """
    Wrapper for the Coco Detection dataset to apply standard transforms
    for input to detection models. Will return the processed image along
    with a tuple of its bounding boxes in ltrb format and labels for each box.

    If a DefaultBoxes object is provided, then will encode the box and labels
    using that object returning a tensor of offsets to the default boxes and
    labels for those boxes and return a three item tuple of the encoded boxes,
    labels, and their original values.

    :param root: The root folder to find the dataset at, if not found will
        download here if download=True
    :param train: True if this is for the training distribution,
        False for the validation
    :param rand_trans: True to apply RandomCrop and RandomHorizontalFlip to the data,
        False otherwise
    :param download: True to download the dataset, False otherwise.
    :param year: The dataset year, supports years 2014, 2015, and 2017.
    :param image_size: the size of the image to output from the dataset
    :param preprocessing_type: Type of standard pre-processing to perform.
        Options are 'yolo', 'ssd', or None.  None defaults to just image normalization
        with no extra processing of bounding boxes.
    :param default_boxes: DefaultBoxes object used to encode bounding boxes and label
        for model loss computation for SSD models. Only used when preprocessing_type=
        'ssd'. Default object represents the default boxes used in standard SSD 300
        implementation.
    """

    def __init__(
        self,
        root: str = default_dataset_path("coco-detection"),
        train: bool = False,
        rand_trans: bool = False,
        download: bool = True,
        year: str = "2017",
        image_size: int = 300,
        preprocessing_type: str = None,
        default_boxes: DefaultBoxes = None,
    ):
        if torchvision_import_error is not None:
            raise torchvision_import_error
        if pycocotools is None:
            raise ValueError("pycocotools is not installed, please install to use")

        if preprocessing_type not in [None, "yolo", "ssd"]:
            raise ValueError(
                "preprocessing type {} not supported, valid values are: {}".format(
                    preprocessing_type, [None, "yolo", "ssd"]
                )
            )

        root = os.path.join(os.path.abspath(os.path.expanduser(root)), str(year))
        if train:
            data_path = "{root}/train{year}".format(root=root, year=year)
            annotation_path = "{root}/annotations/instances_train{year}.json".format(
                root=root, year=year
            )
        else:
            data_path = "{root}/val{year}".format(root=root, year=year)
            annotation_path = "{root}/annotations/instances_val{year}.json".format(
                root=root, year=year
            )

        if not os.path.isdir(data_path) and download:
            dataset_type = "train" if train else "val"
            zip_url = "{COCO_IMAGE_ZIP_ROOT}/{dataset_type}{year}.zip".format(
                COCO_IMAGE_ZIP_ROOT=COCO_IMAGE_ZIP_ROOT,
                dataset_type=dataset_type,
                year=year,
            )
            zip_path = os.path.join(root, "images.zip")
            annotation_url = (
                "{COCO_ANNOTATION_ZIP_ROOT}/annotations_trainval{year}.zip".format(
                    COCO_ANNOTATION_ZIP_ROOT=COCO_ANNOTATION_ZIP_ROOT, year=year
                )
            )
            annotation_zip_path = os.path.join(root, "annotation.zip")
            os.makedirs(root, exist_ok=True)
            print("Downloading coco dataset")

            print("Downloading image files...")
            request.urlretrieve(zip_url, zip_path)
            print("Unzipping image files...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(root)

            print("Downloading annotations files...")
            request.urlretrieve(annotation_url, annotation_zip_path)
            print("Unzipping annotation files...")
            with zipfile.ZipFile(annotation_zip_path, "r") as zip_ref:
                zip_ref.extractall(root)

        elif not os.path.isdir(root):
            raise ValueError(
                f"Coco Dataset Path {root} does not exist. Please download dataset."
            )
        yolo_preprocess = preprocessing_type == "yolo"
        trans = [
            # process annotations
            lambda img, ann: (
                img,
                _extract_bounding_box_and_labels(img, ann, yolo_preprocess),
            ),
        ]
        if rand_trans:
            # add random crop, flip, and jitter to pipeline
            jitter_fn = ColorJitter(
                brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05
            )
            trans.extend(
                [
                    # Random cropping as implemented in SSD paper
                    ssd_random_crop_image_and_annotations,
                    # random horizontal flip
                    random_horizontal_flip_image_and_annotations,
                    # image color jitter
                    lambda img, ann: (jitter_fn(img), ann),
                ]
            )
        trans.extend(
            [
                # resize image
                lambda img, ann: (
                    torch_functional.resize(img, (image_size, image_size)),
                    ann,
                ),
                # Convert image to tensor
                lambda img, ann: (torch_functional.to_tensor(img), ann),
            ]
        )
        # Normalize image except for yolo preprocessing
        if not yolo_preprocess:
            trans.append(
                lambda img, ann: (
                    torch_functional.normalize(
                        img, IMAGENET_RGB_MEANS, IMAGENET_RGB_STDS
                    ),
                    ann,
                )
            )
        if preprocessing_type == "ssd":
            default_boxes = default_boxes or get_default_boxes_300()
            # encode the bounding boxes and labels with the default boxes
            trans.append(
                lambda img, ann: (
                    img,
                    (
                        *default_boxes.encode_image_box_labels(*ann),
                        ann,
                    ),  # encoded_boxes, encoded_labels, original_annotations
                )
            )
        elif yolo_preprocess:
            trans.append(
                lambda img, ann: (
                    img,
                    (bounding_box_and_labels_to_yolo_fmt(ann), ann),
                )
            )

        super().__init__(
            root=data_path,
            annFile=annotation_path,
            transforms=AnnotatedImageTransforms(trans),
        )
        self._default_boxes = default_boxes

    @property
    def default_boxes(self) -> DefaultBoxes:
        """
        :return: The DefaultBoxes object used to encode this datasets bounding boxes
        """
        return self._default_boxes


@DatasetRegistry.register(
    key=["coco_2017_yolo", "coco_detection_yolo", "coco_yolo"],
    attributes={
        "transform_means": IMAGENET_RGB_MEANS,
        "transform_stds": IMAGENET_RGB_STDS,
        "num_classes": 80,
    },
)
def coco_2017_yolo(
    root: str = default_dataset_path("coco-detection"),
    train: bool = False,
    rand_trans: bool = False,
    download: bool = True,
    year: str = "2017",
    image_size: int = 640,
    preprocessing_type: str = "yolo",
):
    """
    Wrapper for COCO detection dataset with Dataset Registry values properly
    created for a Yolo model trained on 80 classes.

    :param root: The root folder to find the dataset at, if not found will
        download here if download=True
    :param train: True if this is for the training distribution,
        False for the validation
    :param rand_trans: True to apply RandomCrop and RandomHorizontalFlip to the data,
        False otherwise
    :param download: True to download the dataset, False otherwise.
    :param year: Only valid option is 2017. default is 2017.
    :param image_size: the size of the image to output from the dataset
    :param preprocessing_type: Type of standard pre-processing to perform.
        Only valid option is 'yolo'. Default is 'yolo'
    """
    if preprocessing_type != "yolo":
        raise ValueError(
            "Only valid preprocessing type for Coco 2017 Yolo dataset is 'yolo'"
            " received: {}".foramt(preprocessing_type)
        )
    if int(year) != 2017:
        raise ValueError(
            "Only valid year type for Coco 2017 Yolo dataset is 2017"
            " received: {}".foramt(year)
        )
    return CocoDetectionDataset(
        root, train, rand_trans, download, year, image_size, "yolo"
    )


def _extract_bounding_box_and_labels(image, annotations, yolo_preprocess=False):
    # returns bounding boxes in ltrb format scaled to [0, 1] and labels
    boxes = []
    labels = []
    for annotation in annotations:
        class_id = int(annotation["category_id"])
        if yolo_preprocess:
            if class_id in _COCO_CLASSES_90_to_80:
                class_id = _COCO_CLASSES_90_to_80[class_id]
            else:
                continue
        boxes.append(annotation["bbox"])
        labels.append(class_id)

    boxes = torch.FloatTensor(boxes)
    labels = torch.Tensor(labels).long()

    if boxes.numel() == 0:
        return boxes, labels

    # convert boxes from ltwh to ltrb
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # r = l + w
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # b = t + h

    # scale boxes to [0, 1]
    boxes[:, [0, 2]] /= image.width  # scale width dimensions
    boxes[:, [1, 3]] /= image.height  # scale height dimensions

    return boxes, labels


# map object ids to class name for post processing
COCO_CLASSES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    12: "street sign",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
    91: "hair brush",
}

# map object ids to class names for 80 class COCO
COCO_CLASSES_80 = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorbike": 3,
    "aeroplane": 4,
    "bus": 5,
    "train": 6,
    "truck": 7,
    "boat": 8,
    "traffic light": 9,
    "fire hydrant": 10,
    "stop sign": 11,
    "parking meter": 12,
    "bench": 13,
    "bird": 14,
    "cat": 15,
    "dog": 16,
    "horse": 17,
    "sheep": 18,
    "cow": 19,
    "elephant": 20,
    "bear": 21,
    "zebra": 22,
    "giraffe": 23,
    "backpack": 24,
    "umbrella": 25,
    "handbag": 26,
    "tie": 27,
    "suitcase": 28,
    "frisbee": 29,
    "skis": 30,
    "snowboard": 31,
    "sports ball": 32,
    "kite": 33,
    "baseball bat": 34,
    "baseball glove": 35,
    "skateboard": 36,
    "surfboard": 37,
    "tennis racket": 38,
    "bottle": 39,
    "wine glass": 40,
    "cup": 41,
    "fork": 42,
    "knife": 43,
    "spoon": 44,
    "bowl": 45,
    "banana": 46,
    "apple": 47,
    "sandwich": 48,
    "orange": 49,
    "broccoli": 50,
    "carrot": 51,
    "hot dog": 52,
    "pizza": 53,
    "donut": 54,
    "cake": 55,
    "chair": 56,
    "sofa": 57,
    "pottedplant": 58,
    "bed": 59,
    "diningtable": 60,
    "toilet": 61,
    "tvmonitor": 62,
    "laptop": 63,
    "mouse": 64,
    "remote": 65,
    "keyboard": 66,
    "cell phone": 67,
    "microwave": 68,
    "oven": 69,
    "toaster": 70,
    "sink": 71,
    "refrigerator": 72,
    "book": 73,
    "clock": 74,
    "vase": 75,
    "scissors": 76,
    "teddy bear": 77,
    "hair drier": 78,
    "toothbrush": 79,
}

# map class ids from 90 class COCO to 80 class COCO
_COCO_CLASSES_90_to_80 = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    13: 11,
    14: 12,
    15: 13,
    16: 14,
    17: 15,
    18: 16,
    19: 17,
    20: 18,
    21: 19,
    22: 20,
    23: 21,
    24: 22,
    25: 23,
    27: 24,
    28: 25,
    31: 26,
    32: 27,
    33: 28,
    34: 29,
    35: 30,
    36: 31,
    37: 32,
    38: 33,
    39: 34,
    40: 35,
    41: 36,
    42: 37,
    43: 38,
    44: 39,
    46: 40,
    47: 41,
    48: 42,
    49: 43,
    50: 44,
    51: 45,
    52: 46,
    53: 47,
    54: 48,
    55: 49,
    56: 50,
    57: 51,
    58: 52,
    59: 53,
    60: 54,
    61: 55,
    62: 56,
    63: 57,
    64: 58,
    65: 59,
    67: 60,
    70: 61,
    72: 62,
    73: 63,
    74: 64,
    75: 65,
    76: 66,
    77: 67,
    78: 68,
    79: 69,
    80: 70,
    81: 71,
    82: 72,
    84: 73,
    85: 74,
    86: 75,
    87: 76,
    88: 77,
    89: 78,
    90: 79,
}
