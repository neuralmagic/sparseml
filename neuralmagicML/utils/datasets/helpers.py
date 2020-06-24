"""
General helper functions for datasets in neuralmagicML
"""

import os


__all__ = [
    "IMAGENET_RGB_MEANS",
    "IMAGENET_RGB_STDS",
    "default_dataset_path",
]


IMAGENET_RGB_MEANS = [0.485, 0.456, 0.406]
IMAGENET_RGB_STDS = [0.229, 0.224, 0.225]


def default_dataset_path(name: str) -> str:
    """
    :param name: name of the dataset to get a path for
    :return: the default path to save the dataset at
    """
    path = os.getenv("NM_ML_DATASETS_PATH", "")

    if not path:
        path = os.path.join("~", ".cache", "nm_datasets")

    path = os.path.join(path, name)

    return path
