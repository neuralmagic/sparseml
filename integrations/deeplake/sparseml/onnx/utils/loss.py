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

import numpy
from scipy.stats import entropy


__all__ = ["kl_divergence"]


def kl_divergence(
    predicted: numpy.ndarray,
    expected: numpy.ndarray,
    zero_point: float = 0.0,
    min_value: float = 1.0,
) -> float:
    """
    Calculate the kl_divergence (entropy) between two input arrays.

    Shifts all values such that the zero_point is at one.
    If a value is lower, then sets it equal to 1.

    :param predicted: the first array to compare with
    :param expected: the second array to compare with
    :param zero_point: the zero point that should be used to shift values above 1
    :param min_value: the minimum value that all values will be truncated to
        if they are below
    :return: the calculated KL divergence
    """
    if predicted.shape != expected.shape:
        raise ValueError(
            "predicted shape of {} must match expected shape of {}".format(
                predicted.shape, expected.shape
            )
        )

    # shift everything to have a min of 1 for the entropy / kl_divergence equation
    predicted = predicted.flatten() - zero_point + min_value
    expected = expected.flatten() - zero_point + min_value

    predicted[predicted < min_value] = min_value
    expected[expected < min_value] = min_value

    divergence = entropy(predicted, expected)

    return divergence
