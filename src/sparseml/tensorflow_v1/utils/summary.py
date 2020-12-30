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
