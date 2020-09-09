"""
Code for working with the tensorflow framework for creating /
editing models for performance in the Neural Magic System
"""

try:
    import tensorflow

    version = [int(v) for v in tensorflow.__version__.split(".")]
    if version[0] != 1 or version[1] < 8:
        raise Exception
except:
    raise RuntimeError(
        "Unable to import tensorflow v1. tensorflow>=1.8,<2.0 is required"
        " to use neuralmagicML.tensorflow."
    )
