import os
import random
from torchvision import transforms
from torchvision.datasets import VOCSegmentation, VOCDetection

from .utils import DATASET_MAPPINGS


__all__ = ['VOCSegmentationDataset', 'VOCDetectionDataset']


class VOCSegmentationDataset(VOCSegmentation):
    def __init__(self, root: str, train: bool = True, rand_trans: bool = False,
                 download: bool = True, year: str = '2012', image_size: int = 300):
        """
        Wrapper for the VOCSegmentation dataset to apply standard transforms

        :param root: The root folder to find the dataset at, if not found will download here if download=True
        :param train: True if this is for the training distribution, false for the validation
        :param rand_trans: True to apply RandomCrop and RandomHorizontalFlip to the data, False otherwise
        :param download: True to download the dataset, False otherwise
                         Base implementation does not support leaving as false if already downloaded
        :param image_size: the size of the image to output from the dataset
        """
        root = os.path.abspath(os.path.expanduser(root))
        trans = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip()
        ] if rand_trans else [transforms.Resize((image_size, image_size))]
        trans.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

        super().__init__(root, year=year, image_set='train' if train else 'val', download=download,
                         transform=transforms.Compose(trans))


DATASET_MAPPINGS['voc_segmentation_2012'] = VOCSegmentationDataset, 21


class VOCDetectionDataset(VOCDetection):
    def __init__(self, root: str, train: bool = True, rand_trans: bool = False,
                 download: bool = True, year: str = '2012', image_size: int = 300):
        """
        Wrapper for the VOCDetection dataset to apply standard transforms

        :param root: The root folder to find the dataset at, if not found will download here if download=True
        :param train: True if this is for the training distribution, false for the validation
        :param rand_trans: True to apply RandomCrop and RandomHorizontalFlip to the data, False otherwise
        :param download: True to download the dataset, False otherwise
                         Base implementation does not support leaving as false if already downloaded
        :param image_size: the size of the image to output from the dataset
        """
        root = os.path.abspath(os.path.expanduser(root))
        trans = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip()
        ] if rand_trans else [transforms.Resize((image_size, image_size))]
        trans.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

        super().__init__(root, year=year, image_set='train' if train else 'val', download=download,
                         transform=transforms.Compose(trans))


DATASET_MAPPINGS['voc_detection_2012'] = VOCDetectionDataset, 21
