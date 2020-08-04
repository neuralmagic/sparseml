"""
Helper classes and functions for PyTorch detection data loaders
"""


from typing import List, Callable


__all__ = [
    "AnnotatedImageTransforms",
]


class AnnotatedImageTransforms(object):
    """
    Class for chaining transforms that take two parameters
    (images and annotations for object detection).

    :param transforms: List of transformations that take an image and annotation as
    their parameters.
    """

    def __init__(self, transforms: List):
        self._transforms = transforms

    @property
    def transforms(self) -> List[Callable]:
        return self._transforms

    def __call__(self, image, annotations):
        for transform in self._transforms:
            image, annotations = transform(image, annotations)
        return image, annotations
