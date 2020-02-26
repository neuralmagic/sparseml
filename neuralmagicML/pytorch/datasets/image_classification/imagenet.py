import os
import random
import PIL.Image as Image

from torchvision import transforms
from torchvision.datasets import ImageFolder

from neuralmagicML.pytorch.datasets.utils import DATASET_MAPPINGS


__all__ = ["ImageNetDataset"]


class ImageNetDataset(ImageFolder):
    RGB_MEANS = [0.485, 0.456, 0.406]
    RGB_STDS = [0.229, 0.224, 0.225]

    def __init__(
        self,
        root: str,
        train: bool = True,
        rand_trans: bool = False,
        image_size: int = 224,
    ):
        """
        Wrapper for the ImageNet dataset to apply standard transforms

        :param root: The root folder to find the dataset at, if not found will download here if download=True
        :param train: True if this is for the training distribution, false for the validation
        :param rand_trans: True to apply RandomCrop and RandomHorizontalFlip to the data, False otherwise
        :param download: True to download the dataset, False otherwise
                         Base implementation does not support leaving as false if already downloaded
        :param image_size: the size of the image to output from the dataset
        """
        non_rand_resize_scale = 256.0 / 224.0  # standard used
        init_trans = (
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
            ]
            if rand_trans
            else [
                transforms.Resize(
                    round(non_rand_resize_scale * image_size),
                    interpolation=Image.BICUBIC,
                ),
                transforms.CenterCrop(image_size),
            ]
        )

        trans = [
            *init_trans,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=ImageNetDataset.RGB_MEANS, std=ImageNetDataset.RGB_STDS
            ),
        ]
        root = os.path.join(
            os.path.abspath(os.path.expanduser(root)), "train" if train else "val"
        )

        super().__init__(root, transform=transforms.Compose(trans))

        if train:
            # make sure we don't preserve the folder structure class order
            random.shuffle(self.samples)


DATASET_MAPPINGS["imagenet"] = ImageNetDataset, 1000
