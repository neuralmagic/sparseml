import os
from PIL import Image
import random
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.datasets import CocoDetection

from neuralmagicML.pytorch.datasets.detection.helpers import AnnotatedImageTransforms
from neuralmagicML.pytorch.datasets.registry import DatasetRegistry
from neuralmagicML.utils.datasets import (
    default_dataset_path,
    IMAGENET_RGB_MEANS,
    IMAGENET_RGB_STDS,
)

import urllib.request as request
import zipfile


__all__ = ["CocoDetectionDataset"]


COCO_IMAGE_ZIP_ROOT = "http://images.cocodataset.org/zips"
COCO_ANNOTATION_ZIP_ROOT = "http://images.cocodataset.org/annotations"


@DatasetRegistry.register(
    key=["coco_detection", "coco"],
    attributes={
        "transform_means": IMAGENET_RGB_MEANS,
        "transform_stds": IMAGENET_RGB_STDS,
    },
)
class CocoDetectionDataset(CocoDetection):
    """
    Wrapper for the Coco Detection dataset to apply standard transforms.

    :param root: The root folder to find the dataset at, if not found will
        download here if download=True
    :param train: True if this is for the training distribution,
        False for the validation
    :param rand_trans: True to apply RandomCrop and RandomHorizontalFlip to the data,
        False otherwise
    :param download: True to download the dataset, False otherwise.
    :param year: The dataset year, supports years 2014, 2015, and 2017.
    :param image_size: the size of the image to output from the dataset
    """

    def __init__(
        self,
        root: str = default_dataset_path("coco-detection"),
        train: bool = False,
        rand_trans: bool = False,
        download: bool = True,
        year: str = "2017",
        image_size: int = 300,
    ):
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

        if not os.path.isdir(root) and download:
            dataset_type = "train" if train else "val"
            zip_url = "{COCO_IMAGE_ZIP_ROOT}/{dataset_type}{year}.zip".format(
                COCO_IMAGE_ZIP_ROOT=COCO_IMAGE_ZIP_ROOT,
                dataset_type=dataset_type,
                year=year,
            )
            zip_path = os.path.join(root, "images.zip")
            annotation_url = "{COCO_ANNOTATION_ZIP_ROOT}/annotations_trainval{year}.zip".format(
                COCO_ANNOTATION_ZIP_ROOT=COCO_ANNOTATION_ZIP_ROOT, year=year
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
                "Coco Dataset Path {root} does not exist. Please download dataset.".format(
                    root=root
                )
            )
        trans = [lambda img, ann: _resize(img, ann, image_size)]
        if rand_trans:
            trans.append(lambda img, ann: _random_horizontal_flip(img, ann))
        trans.extend(
            [
                # Convert to tensor
                lambda img, ann: (F.to_tensor(img), ann),
                # Normalize image
                lambda img, ann: (
                    F.normalize(img, IMAGENET_RGB_MEANS, IMAGENET_RGB_STDS),
                    ann,
                ),
            ]
        )
        super().__init__(
            root=data_path,
            annFile=annotation_path,
            transforms=AnnotatedImageTransforms(trans),
        )


def _resize(image, annotations, image_size):
    if not isinstance(image, Image.Image):
        raise RuntimeError(
            "Loaded image class not supported, expected PIL.Image, got {}".format(
                image.__class__
            )
        )
    width_scale = image_size / image.size[0]
    height_scale = image_size / image.size[1]

    def update_bbox_fn(bbox):
        return [
            bbox[0] * width_scale,
            bbox[1] * height_scale,
            bbox[2] * width_scale,
            bbox[3] * height_scale,
        ]

    image = F.resize(image, (image_size, image_size))
    annotations = _update_bbox_values(annotations, update_bbox_fn)

    return image, annotations


def _random_horizontal_flip(image, annotations, p=0.5):
    if not isinstance(image, Image.Image):
        raise RuntimeError(
            "Loaded image class not supported, expected PIL.Image, got {}".format(
                image.__class__
            )
        )

    if random.random() < p:

        def bbox_horizontal_flip_fn(bbox):
            bbox[0] = image.size[0] - bbox[0] - bbox[2]
            return bbox

        image = F.hflip(image)
        annotations = _update_bbox_values(annotations, bbox_horizontal_flip_fn)

    return image, annotations


def _update_bbox_values(annotations, update_bbox_fn):
    for idx, annotation in enumerate(annotations):
        updated_bbox = update_bbox_fn(annotation["bbox"])
        annotations[idx]["bbox"] = updated_bbox
    return annotations
