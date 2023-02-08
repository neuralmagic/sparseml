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
Convenience functions for tensorboard and writing summaries to it
"""

from typing import Any

from sparseml.tensorflow_v1.utils.helpers import tf_compat


__all__ = ["write_simple_summary"]


def write_simple_summary(
    writer: tf_compat.summary.FileWriter, tag: str, val: Any, step: int
):
    """
    Write a simple value summary to a writer

    :param writer: the writer to write the summary to
    :param tag: the tag to write the value under
    :param val: the value to write
    :param step: the current global step to write the value at
    """
    value = tf_compat.Summary.Value(tag=tag, simple_value=val)
    summary = tf_compat.Summary(value=[value])
    writer.add_summary(summary, step)
