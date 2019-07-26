import os
import random
from torchvision import transforms
from torchvision.datasets import ImageNet


__all__ = ['ImageNetDataset']


class ImageNetDataset(ImageNet):
    def __init__(self, root: str, train: bool = True, rand_trans: bool = False,
                 download: bool = False, image_size: int = 224):
        """
        Wrapper for the ImageNet dataset to apply standard transforms

        :param root: The root folder to find the dataset at, if not found will download here if download=True
        :param train: True if this is for the training distribution, false for the validation
        :param rand_trans: True to apply RandomCrop and RandomHorizontalFlip to the data, False otherwise
        :param download: True to download the dataset, False otherwise
                         Base implementation does not support leaving as false if already downloaded
        :param image_size: the size of the image to output from the dataset
        """
        root = os.path.abspath(os.path.expanduser(root))
        trans = [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip()
        ] if rand_trans else [transforms.CenterCrop(image_size)]
        trans.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        super().__init__(root, split='train' if train else 'val', download=download,
                         transform=transforms.Compose(trans))

        # make sure we don't preserve the folder structure class order
        random.shuffle(self.samples)
