import tensorflow as tf


__all__ = [
    "tf_compat",
    "tf_compat_div",
]


tf_compat = (
    tf
    if not hasattr(tf, "compat") or not hasattr(getattr(tf, "compat"), "v1")
    else tf.compat.v1
)  # type: tf
tf_compat_div = (
    tf_compat.div
    if not hasattr(tf_compat, "math")
    or not hasattr(getattr(tf_compat, "math"), "divide")
    else tf_compat.math.divide
)
