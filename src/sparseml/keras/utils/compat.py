try:
    import keras as native_keras
except ModuleNotFoundError:
    native_keras = None

import tensorflow


__all__ = [
    "assign",
    "keras",
]


keras = native_keras if native_keras is not None else tensorflow.keras


def assign(lhs, rhs, name=None):
    if hasattr(tensorflow, "assign"):
        return tensorflow.assign(lhs, rhs, name=name)
    else:
        return lhs.assign(rhs, name=name)
