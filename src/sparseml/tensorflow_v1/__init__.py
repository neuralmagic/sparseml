"""
Code for working with the tensorflow_v1 framework for creating /
editing models for performance in the Neural Magic System
"""

try:
    import tensorflow

    version = [int(v) for v in tensorflow.__version__.split(".")]
    if version[0] != 1 or version[1] < 8:
        raise Exception
except:
    raise RuntimeError(
        "Unable to import tensorflow_v1 v1. tensorflow_v1>=1.8,<2.0 is required"
        " to use sparseml.tensorflow_v1_v1."
    )
