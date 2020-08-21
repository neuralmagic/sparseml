"""
Helper functions to edit ONNX Graphs.
"""


import numpy as np
import onnx

from onnx import numpy_helper, ModelProto


__all__ = [
    "update_model_param",
    "swap_node_output",
    "remove_node_and_params_from_graph",
    "override_model_batch_size",
]


def update_model_param(model: ModelProto, param_name: str, val: np.ndarray,) -> None:
    """
    Removes the parameter with name param_name from the model
    Creates a new parameter using val
    Adds val to the model with name param_name as an update
    :param model: The model to update
    :param param_name: The parameter name in the model to update
    :param val: The new value of the parameter
    """
    param_matches = [
        param for param in model.graph.initializer if param.name == param_name
    ]
    if param_matches:
        model.graph.initializer.remove(param_matches[0])
    new_param = numpy_helper.from_array(val, param_name)
    model.graph.initializer.append(new_param)


def swap_node_output(node: onnx.NodeProto, output: str) -> None:
    """
    Deletes the current output of the node and replaces it with the provided value
    Assumes that the node only has one output
    :param node: Node to change the output of
    :param output: New output value
    """
    node.output.pop()
    node.output.append(output)


def remove_node_and_params_from_graph(model: ModelProto, node: onnx.NodeProto) -> None:
    """
    Deletes a node from the mdoel graph as well as its parameters listed in node.input
    :param model: Model to delete from
    :param node: Node to delete
    """
    param_matches = [p for p in model.graph.initializer if p.name in node.input]
    for param in param_matches:
        model.graph.initializer.remove(param)
    model.graph.node.remove(node)


def _override_tensor_batch_dim(model, tensor, batch_size):
    for init in model.graph.initializer:
        if init.name == tensor.name:
            # This tensor is actually an initializer => skip
            return

    shape = tensor.type.tensor_type.shape

    # skip tensors with variable batch sizes
    if not shape.dim[0].dim_param and shape.dim[0].dim_value > 0:
        shape.dim[0].dim_value = batch_size


def override_model_batch_size(model: ModelProto, batch_size: int) -> ModelProto:
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
