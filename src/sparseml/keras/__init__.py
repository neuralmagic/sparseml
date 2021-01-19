"""
Code for working with the keras framework for creating /
editing models for performance in the Neural Magic System
"""

try:
    import tensorflow

    version = [int(v) for v in tensorflow.__version__.split(".")]
    if version[0] != 2 or version[1] < 2:
        raise Exception
except:
    raise RuntimeError(
        "Unable to import tensorflow. tensorflow>=2.2 is required"
        " to use sparseml.keras."
    )
