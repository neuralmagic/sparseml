# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
General utilities for dataset implementations for TensorFlow
"""

from typing import Tuple

from sparseml.tensorflow_v1.utils import tf_compat


__all__ = [
    "resize",
    "random_scaling_crop",
    "center_square_crop",
]


def resize(image_size: Tuple[int, int], name: str = "resize"):
    """
    Resize an image tensor to the desired size

    :param image_size: a tuple containing the height, width to resize to
    :param name: name for the scope to put the ops under
    :return: the callable function for resize op, takes in the image and outputs
        the resized image
    """

    def res(img: tf_compat.Tensor):
        with tf_compat.name_scope(name):
            try:
                img = tf_compat.image.resize(img, image_size)
            except Exception:
                img = tf_compat.image.resize_images(img, image_size)

        return img

    return res


def random_scaling_crop(
    scale_range: Tuple[int, int] = (0.08, 1.0),
    ratio_range: Tuple[int, int] = (3.0 / 4.0, 4.0 / 3.0),
    name: str = "random_scaling_crop",
):
    """
    Random crop implementation which also randomly scales the crop taken
    as well as the aspect ratio of the crop.

    :param scale_range: the (min, max) of the crop scales to take from the orig image
    :param ratio_range: the (min, max) of the aspect ratios to take from the orig image
    :param name: name for the scope to put the ops under
    :return: the callable function for random scaling crop op,
        takes in the image and outputs randomly cropped image
    """

    def rand_crop(img: tf_compat.Tensor):
        with tf_compat.name_scope(name):
            orig_shape = tf_compat.shape(img)
            scale = tf_compat.random_uniform(
                shape=[1], minval=scale_range[0], maxval=scale_range[1]
            )[0]
            ratio = tf_compat.random_uniform(
                shape=[1], minval=ratio_range[0], maxval=ratio_range[1]
            )[0]
            height = tf_compat.minimum(
                tf_compat.cast(
                    tf_compat.round(
                        tf_compat.cast(orig_shape[0], dtype=tf_compat.float32)
                        * scale
                        / ratio
                    ),
                    tf_compat.int32,
                ),
                orig_shape[0],
            )
            width = tf_compat.minimum(
                tf_compat.cast(
                    tf_compat.round(
                        tf_compat.cast(orig_shape[1], dtype=tf_compat.float32) * scale
                    ),
                    tf_compat.int32,
                ),
                orig_shape[1],
            )
            img = tf_compat.random_crop(img, [height, width, orig_shape[2]])

            return img

    return rand_crop


def center_square_crop(padding: int = 0, name: str = "center_square_crop"):
    """
    Take a square crop centered in the a image

    :param padding: additional padding to apply to all sides of the image
        to crop away
    :param name: name for the scope to put the ops under
    :return: the callable function for square crop op,
        takes in the image and outputs the cropped image
    """

    def cent_crop(img: tf_compat.Tensor):
        with tf_compat.name_scope(name):
            orig_shape = tf_compat.shape(img)
            min_size = tf_compat.cond(
                tf_compat.greater_equal(orig_shape[0], orig_shape[1]),
                lambda: orig_shape[1],
                lambda: orig_shape[0],
            )

            if padding > 0:
                orig_shape_list = img.get_shape().as_list()
                resize(
                    (orig_shape_list[0] + 2 * padding, orig_shape_list[1] + 2 * padding)
                )

            padding_height = tf_compat.add(
                tf_compat.cast(
                    tf_compat.round(
                        tf_compat.div(
                            tf_compat.cast(
                                tf_compat.subtract(orig_shape[0], min_size),
                                tf_compat.float32,
                            ),
                            2.0,
                        )
                    ),
                    tf_compat.int32,
                ),
                padding,
            )
            padding_width = tf_compat.add(
                tf_compat.cast(
                    tf_compat.round(
                        tf_compat.div(
                            tf_compat.cast(
                                tf_compat.subtract(orig_shape[1], min_size),
                                tf_compat.float32,
                            ),
                            2.0,
                        )
                    ),
                    tf_compat.int32,
                ),
                padding,
            )
            img = tf_compat.image.crop_to_bounding_box(
                img, padding_height, padding_width, min_size, min_size
            )

            return img

    return cent_crop
