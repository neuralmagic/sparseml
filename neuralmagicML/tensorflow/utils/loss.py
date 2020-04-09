from neuralmagicML.tensorflow.utils.helpers import tf_compat


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
