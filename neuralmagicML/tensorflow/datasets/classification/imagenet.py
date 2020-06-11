"""
Imagenet dataset implementations for the image classification field in computer vision.
More info for the dataset can be found `here <http://www.image-net.org/>`__.
"""
import os

from neuralmagicML.utils.datasets import (
    default_dataset_path,
    IMAGENET_RGB_MEANS,
    IMAGENET_RGB_STDS,
)
from neuralmagicML.tensorflow.utils import tf_compat, tf_compat_div
from neuralmagicML.tensorflow.datasets.registry import DatasetRegistry
from neuralmagicML.tensorflow.datasets.dataset import (
    ImageFolderDataset,
    random_scaling_crop,
    center_square_crop,
)

__all__ = ["ImageNetDataset", "imagenet_normalizer"]


def imagenet_normalizer(img):
    """
    Normalize an image using mean and std of the imagenet dataset

    :param img: The input image to normalize
    :return: The normalized image
    """
    img = tf_compat_div(img, 255.0)
    means = tf_compat.constant(IMAGENET_RGB_MEANS, dtype=tf_compat.float32)
    stds = tf_compat.constant(IMAGENET_RGB_STDS, dtype=tf_compat.float32)
    img = tf_compat_div(tf_compat.subtract(img, means), stds)
    return img


@DatasetRegistry.register(
    key=["imagenet"],
    attributes={
        "num_classes": 1000,
        "transform_means": IMAGENET_RGB_MEANS,
        "transform_stds": IMAGENET_RGB_STDS,
    },
)
class ImageNetDataset(ImageFolderDataset):
    """
    ImageNet dataset implementation

    :param root: The root folder to find the dataset at, if not found will
        download here if download=True
    :param train: True if this is for the training distribution,
        False for the validation
    :param rand_trans: True to apply RandomCrop and RandomHorizontalFlip to the data,
        False otherwise
    :param image_size: the size of the image to output from the dataset
    """

    def __init__(
        self,
        root: str = default_dataset_path("imagenet"),
        train: bool = True,
        rand_trans: bool = False,
        image_size: int = 224,
    ):
        self._train = train

        if rand_trans:
            transforms = [
                random_scaling_crop(),
                tf_compat.image.random_flip_left_right,
                tf_compat.image.random_flip_up_down,
            ]
        else:
            transforms = [center_square_crop()]

        root = os.path.join(root, "train") if train else os.path.join(root, "val")

        super().__init__(
            root, image_size, transforms, imagenet_normalizer,
        )

    def name_scope(self) -> str:
        """
        :return: the name scope the dataset should be built under in the graph
        """
        return "Imagenet_{}".format("train" if self._train else "val")
