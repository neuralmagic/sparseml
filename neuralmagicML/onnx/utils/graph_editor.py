"""
Helper functions to edit ONNX Graphs.
"""


import onnx

from onnx import ModelProto


__all__ = [
    "override_model_batch_size",
]


def _override_tensor_batch_dim(model, tensor, batch_size):
    for init in model.graph.initializer:
        if init.name == tensor.name:
            # This tensor is actually an initializer => skip
            return

    shape = tensor.type.tensor_type.shape

    # skip tensors with variable batch sizes
    if not shape.dim[0].dim_param and shape.dim[0].dim_value > 0:
        shape.dim[0].dim_value = batch_size


def override_model_batch_size(model: ModelProto, batch_size: int):
    """
    Rewrites any positive batch dimensions in the model inputs or outputs to the
    given batch_size

    :param model: Model to modify
    :param batch_size: Batch size to enforce
    :return: the given model with inputs and outputs set to batch_size if the batch
        dimensions are not -1.
    """
    for tensor in model.graph.input:
        # This may not work for ONNX graphs that have hard-coded reshape nodes
        _override_tensor_batch_dim(model, tensor, batch_size)
    # Do the same for outputs
    for tensor in model.graph.output:
        # Ignore augmented _Reduce nodes
        if "_Reduce" not in tensor.name:
            _override_tensor_batch_dim(model, tensor, batch_size)
