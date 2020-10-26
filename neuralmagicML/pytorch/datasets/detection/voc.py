"""
VOC dataset implementations for the object detection field in computer vision.
More info for the dataset can be found
`here <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/
PascalVOC_IJCV2009.pdf>`__.
"""

import os
from PIL import Image
import random
import torch
from torchvision import transforms
from torchvision.transforms import (
    ColorJitter,
    functional as F,
)

try:
    from torchvision.datasets import VOCSegmentation, VOCDetection
except ModuleNotFoundError:
    # older version of pytorch, VOC not available
    VOCSegmentation = object
    VOCDetection = object

from neuralmagicML.pytorch.datasets.detection.helpers import (
    AnnotatedImageTransforms,
    ssd_random_crop_image_and_annotations,
    random_horizontal_flip_image_and_annotations,
    bounding_box_and_labels_to_yolo_fmt,
)
from neuralmagicML.pytorch.datasets.registry import DatasetRegistry
from neuralmagicML.pytorch.utils import (
    DefaultBoxes,
    get_default_boxes_300,
)
from neuralmagicML.utils.datasets import (
    default_dataset_path,
    IMAGENET_RGB_MEANS,
    IMAGENET_RGB_STDS,
)


__all__ = [
    "VOCSegmentationDataset",
    "VOCDetectionDataset",
    "VOC_CLASSES",
]


@DatasetRegistry.register(
    key=["voc_seg", "voc_segmentation"],
    attributes={
        "num_classes": 21,
        "transform_means": IMAGENET_RGB_MEANS,
        "transform_stds": IMAGENET_RGB_STDS,
    },
)
class VOCSegmentationDataset(VOCSegmentation):
    """
    Wrapper for the VOC Segmentation dataset to apply standard transforms.

    :param root: The root folder to find the dataset at, if not found will
        download here if download=True
    :param train: True if this is for the training distribution,
        False for the validation
    :param rand_trans: True to apply RandomCrop and RandomHorizontalFlip to the data,
        False otherwise
    :param download: True to download the dataset, False otherwise.
    :param year: The dataset year, supports years 2007 to 2012.
    :param image_size: the size of the image to output from the dataset
    """

    def __init__(
        self,
        root: str = default_dataset_path("voc-segmentation"),
        train: bool = True,
        rand_trans: bool = False,
        download: bool = True,
        year: str = "2012",
        image_size: int = 300,
    ):
        if VOCSegmentation is object:
            raise ValueError(
                "VOC is unsupported on this PyTorch version, please upgrade to use"
            )

        root = os.path.abspath(os.path.expanduser(root))
        trans = (
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
            ]
            if rand_trans
            else [transforms.Resize((image_size, image_size))]
        )
        trans.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_RGB_MEANS, std=IMAGENET_RGB_STDS),
            ]
        )

        super().__init__(
            root,
            year=year,
            image_set="train" if train else "val",
            download=download,
            transform=transforms.Compose(trans),
        )


@DatasetRegistry.register(
    key=["voc_det", "voc_detection"],
    attributes={
        "num_classes": 21,
        "transform_means": IMAGENET_RGB_MEANS,
        "transform_stds": IMAGENET_RGB_STDS,
    },
)
class VOCDetectionDataset(VOCDetection):
    """
    Wrapper for the VOC Detection dataset to apply standard transforms
    for input to detection models. Will return the processed image along
    with a tuple of its bounding boxes in ltrb format and labels for each box.

    If a DefaultBoxes object is provided, then will encode the box and labels
    using that object returning a tensor of offsets to the default boxes and
    labels for those boxes and return a three item tuple of the encoded boxes,
    labels, and their original values.

    :param root: The root folder to find the dataset at, if not found will download
        here if download=True
    :param train: True if this is for the training distribution,
        False for the validation
    :param rand_trans: True to apply RandomCrop and RandomHorizontalFlip to the data,
        False otherwise
    :param download: True to download the dataset, False otherwise.
        Base implementation does not support leaving as false if already downloaded
    :param image_size: the size of the image to output from the dataset
    :param preprocessing_type: Type of standard pre-processing to perform.
        Options are 'yolo', 'ssd', or None.  None defaults to just image normalization
        with no extra processing of boudning boxes.
    :param default_boxes: DefaultBoxes object used to encode bounding boxes and label
        for model loss computation for SSD models. Only used when preprocessing_type=
        'ssd'. Default object represents the default boxes used in standard SSD 300
        implementation.
    """

    def __init__(
        self,
        root: str = default_dataset_path("voc-detection"),
        train: bool = True,
        rand_trans: bool = False,
        download: bool = True,
        year: str = "2012",
        image_size: int = 300,
        preprocessing_type: str = None,
        default_boxes: DefaultBoxes = None,
    ):
        if VOCDetection == object:
            raise ValueError(
                "VOC is unsupported on this PyTorch version, please upgrade to use"
            )
        if preprocessing_type not in [None, "yolo", "ssd"]:
            raise ValueError(
                "preprocessing type {} not supported, valid values are: {}".format(
                    preprocessing_type, [None, "yolo", "ssd"]
                )
            )

        root = os.path.abspath(os.path.expanduser(root))
        trans = [
            # process annotations
            lambda img, ann: (img, _extract_bounding_box_and_labels(img, ann)),
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
                lambda img, ann: (F.resize(img, (image_size, image_size)), ann),
                # Convert image to tensor
                lambda img, ann: (F.to_tensor(img), ann),
            ]
        )
        # Normalize image except for yolo preprocessing
        if preprocessing_type != "yolo":
            trans.append(
                lambda img, ann: (
                    F.normalize(img, IMAGENET_RGB_MEANS, IMAGENET_RGB_STDS),
                    ann,
                )
            )

        if preprocessing_type == "ssd":
            default_boxes = default_boxes or get_default_boxes_300(voc=True)
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
        if preprocessing_type == "yolo":
            trans.append(
                lambda img, ann: (img, (bounding_box_and_labels_to_yolo_fmt(ann), ann),)
            )
        super().__init__(
            root,
            year=year,
            image_set="train" if train else "val",
            download=download,
            transforms=AnnotatedImageTransforms(trans),
        )
        self._default_boxes = default_boxes

    @property
    def default_boxes(self) -> DefaultBoxes:
        """
        :return: The DefaultBoxes object used to encode this datasets bounding boxes
        """
        return self._default_boxes


def _extract_bounding_box_and_labels(image, annotations):
    # returns bounding boxes in ltrb format scaled to [0, 1] and labels
    boxes = []
    labels = []
    box_objects = annotations["annotation"]["object"]
    if isinstance(box_objects, dict):
        box_objects = [box_objects]
    for annotation in box_objects:
        boxes.append(
            [
                float(annotation["bndbox"]["xmin"]),
                float(annotation["bndbox"]["ymin"]),
                float(annotation["bndbox"]["xmax"]),
                float(annotation["bndbox"]["ymax"]),
            ]
        )
        labels.append(_VOC_CLASS_NAME_TO_ID[annotation["name"]])

    boxes = torch.Tensor(boxes).float()
    labels = torch.Tensor(labels).long()

    # scale boxes to [0, 1]
    boxes[:, [0, 2]] /= image.width  # scale width dimensions
    boxes[:, [1, 3]] /= image.height  # scale height dimensions

    return boxes, labels


_VOC_CLASS_NAME_TO_ID = {
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20,
}

# Map class ids to names for model post processing
VOC_CLASSES = {(v, k) for (k, v) in _VOC_CLASS_NAME_TO_ID.items()}
