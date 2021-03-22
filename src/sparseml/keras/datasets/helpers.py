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
General utilities for dataset implementations for Keras
"""

from typing import Tuple

import tensorflow


__all__ = [
    "random_scaling_crop",
]


def random_scaling_crop(
    scale_range: Tuple[int, int] = (0.8, 1.0),
    ratio_range: Tuple[int, int] = (3.0 / 4.0, 4.0 / 3.0),
):
    """
    Random crop implementation which also randomly scales the crop taken
    as well as the aspect ratio of the crop.
    :param scale_range: the (min, max) of the crop scales to take from the orig image
    :param ratio_range: the (min, max) of the aspect ratios to take from the orig image
    :return: the callable function for random scaling crop op,
        takes in the image and outputs randomly cropped image
    """

    def rand_crop(img: tensorflow.Tensor):
        orig_shape = tensorflow.shape(img)
        scale = tensorflow.random.uniform(
            shape=[1], minval=scale_range[0], maxval=scale_range[1]
        )[0]
        ratio = tensorflow.random.uniform(
            shape=[1], minval=ratio_range[0], maxval=ratio_range[1]
        )[0]
        height = tensorflow.minimum(
            tensorflow.cast(
                tensorflow.round(
                    tensorflow.cast(orig_shape[0], dtype=tensorflow.float32)
                    * scale
                    / ratio
                ),
                tensorflow.int32,
            ),
            orig_shape[0],
        )
        width = tensorflow.minimum(
            tensorflow.cast(
                tensorflow.round(
                    tensorflow.cast(orig_shape[1], dtype=tensorflow.float32) * scale
                ),
                tensorflow.int32,
            ),
            orig_shape[1],
        )
        img = tensorflow.image.random_crop(img, [height, width, orig_shape[2]])

        return img

    return rand_crop
