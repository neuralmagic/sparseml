from typing import Union

from neuralmagicML.utils.datasets import (
    default_dataset_path,
    ImagenetteDownloader,
    ImagewoofDownloader,
    ImagenetteSize,
    IMAGENET_RGB_MEANS,
    IMAGENET_RGB_STDS,
)
from neuralmagicML.tensorflow.utils import tf_compat
from neuralmagicML.tensorflow.datasets.registry import DatasetRegistry
from neuralmagicML.tensorflow.datasets.dataset import (
    random_scaling_crop,
    center_square_crop,
    ImageFolderDataset,
)

from neuralmagicML.tensorflow.datasets.classification import imagenet_normalizer

__all__ = ["ImagenetteSize", "ImagenetteDataset", "ImagewoofDataset"]


@DatasetRegistry.register(
    key=["imagenette"],
    attributes={
        "num_classes": 10,
        "transform_means": IMAGENET_RGB_MEANS,
        "transform_stds": IMAGENET_RGB_STDS,
    },
)
class ImagenetteDataset(ImageFolderDataset, ImagenetteDownloader):
    """
    Wrapper for the imagenette (10 class) dataset that fastai created.
    Handles downloading and applying standard transforms.

    :param root: The root folder to find the dataset at,
        if not found will download here if download=True
    :param train: True if this is for the training distribution,
        False for the validation
    :param rand_trans: True to apply RandomCrop and RandomHorizontalFlip to the data,
        False otherwise
    :param dataset_size: The size of the dataset to use and download:
        See ImagenetteSize for options
    :param image_size: The image size to output from the dataset
    :param download: True to download the dataset, False otherwise
    """

    def __init__(
        self,
        root: str = default_dataset_path("imagenette"),
        train: bool = True,
        rand_trans: bool = False,
        dataset_size: ImagenetteSize = ImagenetteSize.s160,
        image_size: Union[int, None] = None,
        download: bool = True,
    ):
        ImagenetteDownloader.__init__(self, root, dataset_size, download)
        self._train = train

        if image_size is None:
            image_size = 160 if dataset_size == ImagenetteSize.s160 else 224

        if rand_trans:
            transforms = [
                random_scaling_crop(),
                tf_compat.image.random_flip_left_right,
                tf_compat.image.random_flip_up_down,
            ]
        else:
            transforms = [center_square_crop()]

        super().__init__(
            self.split_root(train), image_size, transforms, imagenet_normalizer
        )

    def name_scope(self) -> str:
        """
        :return: the name scope the dataset should be built under in the graph
        """
        return "Imagenette_{}".format("train" if self._train else "val")


@DatasetRegistry.register(
    key=["imagewoof"],
    attributes={
        "num_classes": 10,
        "transform_means": IMAGENET_RGB_MEANS,
        "transform_stds": IMAGENET_RGB_STDS,
    },
)
class ImagewoofDataset(ImageFolderDataset, ImagewoofDownloader):
    """
    Wrapper for the imagewoof (10 class) dataset that fastai created.
    Handles downloading and applying standard transforms.
    More info for the dataset can be found `here <https://github.com/fastai/imagenette>`

    :param root: The root folder to find the dataset at,
        if not found will download here if download=True
    :param train: True if this is for the training distribution,
        False for the validation
    :param rand_trans: True to apply RandomCrop and RandomHorizontalFlip to the data,
        False otherwise
    :param dataset_size: The size of the dataset to use and download:
        See :py:func `~ImagewoofSize` for options
    :param image_size: The image size to output from the dataset
    :param download: True to download the dataset, False otherwise
    """

    def __init__(
        self,
        root: str = default_dataset_path("imagewoof"),
        train: bool = True,
        rand_trans: bool = False,
        dataset_size: ImagenetteSize = ImagenetteSize.s160,
        image_size: Union[int, None] = None,
        download: bool = True,
    ):
        ImagewoofDownloader.__init__(self, root, dataset_size, download)
        self._train = train

        if image_size is None:
            image_size = 160 if dataset_size == ImagenetteSize.s160 else 224

        if rand_trans:
            transforms = [
                random_scaling_crop(),
                tf_compat.image.random_flip_left_right,
                tf_compat.image.random_flip_up_down,
            ]
        else:
            transforms = [center_square_crop()]

        super().__init__(
            self.split_root(train), image_size, transforms, imagenet_normalizer
        )

    def name_scope(self) -> str:
        """
        :return: the name scope the dataset should be built under in the graph
        """
        return "Imagenette_{}".format("train" if self._train else "val")
