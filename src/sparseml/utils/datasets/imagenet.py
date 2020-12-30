"""
General utilities for the imagenet dataset implementations for the
image classification field in computer vision.
More info for the dataset can be found `here <http://www.image-net.org/>`__.
"""

__all__ = [
    "IMAGENET_RGB_MEANS",
    "IMAGENET_RGB_STDS",
]


IMAGENET_RGB_MEANS = [0.485, 0.456, 0.406]
IMAGENET_RGB_STDS = [0.229, 0.224, 0.225]
