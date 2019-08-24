from typing import Union
import os
from enum import Enum
import random
import tarfile
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url

from .utils import DATASET_MAPPINGS


__all__ = ['ImagewoofSize', 'ImagewoofDataset']


class ImagewoofSize(Enum):
    full = 'full'
    s320 = 's320'
    s160 = 's160'


class ImagewoofDataset(ImageFolder):
    def __init__(self, root: str, train: bool = True, rand_trans: bool = False,
                 dataset_size: ImagewoofSize = ImagewoofSize.s160, image_size: Union[int, None] = None,
                 download: bool = True):
        """
        Wrapper for the imagewoof (10 class) dataset that fastai created
        Handles downloading and applying standard transforms

        :param root: The root folder to find the dataset at, if not found will download here if download=True
        :param train: True if this is for the training distribution, false for the validation
        :param rand_trans: True to apply RandomCrop and RandomHorizontalFlip to the data, False otherwise
        :param dataset_size: The size of the dataset to use and download: See ImagenetteSize for options
        :param image_size: The image size to output from the dataset
        :param download: True to download the dataset, False otherwise
        """
        root = os.path.abspath(os.path.expanduser(root))

        if not os.path.exists(root):
            os.makedirs(root)

        extracted_root = root

        if download:
            if dataset_size == ImagewoofSize.s160:
                extract = 'imagewoof-160'
            elif dataset_size == ImagewoofSize.s320:
                extract = 'imagewoof-320'
            elif dataset_size == ImagewoofSize.full:
                extract = 'imagewoof'
            else:
                raise Exception('Unknown ImagewoofSize given of {}'.format(dataset_size))

            extracted_root = os.path.join(root, extract)
            self.download(dataset_size, root, extract)

        if image_size is None:
            image_size = 160 if dataset_size == ImagewoofSize.s160 else 224

        extracted_root = os.path.join(extracted_root, 'train') if train else os.path.join(extracted_root, 'val')
        trans = [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip()
        ] if rand_trans else [transforms.CenterCrop(image_size)]
        trans.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        super().__init__(extracted_root, transforms.Compose(trans))

        # make sure we don't preserve the folder structure class order
        random.shuffle(self.samples)

    @staticmethod
    def download(size: ImagewoofSize, root: str, extract: str):
        if size == ImagewoofSize.full:
            url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagewoof.tgz'
        elif size == ImagewoofSize.s320:
            url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagewoof-320.tgz'
        elif size == ImagewoofSize.s160:
            url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagewoof-160.tgz'
        else:
            raise Exception('unknown imagewoof size given of {}'.format(size))

        filename = '{}.tar'.format(extract)

        if os.path.exists(os.path.join(root, filename)):
            print('already downloaded imagewoof of size {}'.format(size))

            return

        download_url(url, root, filename)

        with tarfile.open(os.path.join(root, filename), 'r:gz') as tar:
            tar.extractall(path=root)


DATASET_MAPPINGS['imagewoof'] = ImagewoofDataset, 10
