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

from sparseml.tensorflow_v1.utils.helpers import tf_compat


__all__ = ["batch_cross_entropy_loss", "accuracy"]


def batch_cross_entropy_loss(
    logits: tf_compat.Tensor, labels: tf_compat.Tensor
) -> tf_compat.Tensor:
    """
    Standard cross entropy loss that reduces across the batch dimension.

    :param logits: the logits from the model to use
    :param labels: the labels to compare the logits to
    :return: the cross entropy loss
    """
    with tf_compat.name_scope("loss/batch_cross_entropy/"):
        return tf_compat.reduce_mean(
            tf_compat.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=labels
            )
        )


def accuracy(
    logits: tf_compat.Tensor, labels: tf_compat.Tensor, index: int = 1
) -> tf_compat.Tensor:
    """
    Standard evaluation for accuracy.

    :param logits: the logits from the model to use
    :param labels: the labels to compare the logits to
    :param index: the index in the tensors to compare against
    :return: the accuracy
    """
    with tf_compat.name_scope("loss/accuracy/"):
        return tf_compat.reduce_mean(
            tf_compat.cast(
                tf_compat.equal(
                    tf_compat.argmax(logits, index), tf_compat.argmax(labels, index)
                ),
                tf_compat.float32,
            )
        )
