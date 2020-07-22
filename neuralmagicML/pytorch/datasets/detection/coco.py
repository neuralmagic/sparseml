import os
from torchvision import transforms
from torchvision.datasets import CocoDetection

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
                COCO_ANNOTATION_ZIP_ROOT=COCO_ANNOTATION_ZIP_ROOT,
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
            root=data_path, annFile=annotation_path, transform=transforms.Compose(trans)
        )
