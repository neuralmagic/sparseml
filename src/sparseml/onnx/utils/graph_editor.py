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

"""
Helper functions to edit ONNX Graphs.
"""

from typing import Iterable, List, Union

import numpy
import onnx
from onnx import ModelProto, NodeProto, numpy_helper

from sparseml.onnx.utils.helpers import get_node_params


__all__ = [
    "update_model_param",
    "swap_node_output",
    "remove_node_and_params_from_graph",
    "override_model_batch_size",
    "prune_unstructured",
    "prune_model_one_shot",
    "prune_model_one_shot_iter",
]


def update_model_param(
    model: ModelProto,
    param_name: str,
    val: numpy.ndarray,
) -> None:
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


def remove_node_and_params_from_graph(
    model: ModelProto,
    node: onnx.NodeProto,
    keep_params: Iterable[str] = None,
) -> None:
    """
    Deletes a node from the mdoel graph as well as its parameters listed in node.input

    :param model: Model to delete from
    :param node: Node to delete
    :param keep_params: Names of node input initializers not to remove from graph
        default is None.
    """
    keep_params = keep_params or []
    for param in model.graph.initializer:
        if param.name not in keep_params and param.name in node.input:
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


def prune_unstructured(array: numpy.ndarray, sparsity: float) -> numpy.ndarray:
    """
    Prune a numpy array with unstructured sparsity according to magnitude pruning

    :param array: the array to prune (introduce zeros), will remove the lowest
        absolute values in the array
    :param sparsity: the sparsity value, as a decimal, to impose in the array
    :return: the pruned array
    """
    array = numpy.array(array)  # make a copy because arrays from onnx are read only
    sparse_index = int(round(sparsity * array.size) - 1)

    if sparse_index < 0:
        return array

    sorted_array = numpy.sort(numpy.abs(array.flatten()))
    sparse_thresh = sorted_array[sparse_index]
    array[numpy.abs(array) < sparse_thresh] = 0

    return array


def prune_model_one_shot(
    model: ModelProto, nodes: List[NodeProto], sparsity: Union[float, List[float]]
):
    """
    Prune a model in-place with one shot pruning (no retraining) according to
    magnitude pruning. Does so in an unstructured way currently

    :param model: the model to apply pruning to
    :param nodes: the nodes within the model to prune to the desired sparsities
    :param sparsity: the sparsity level to prune all nodes to if a float,
        or the sparsity level to prune each node to if a list of floats
    :return: the new, pruned model
    """
    if not isinstance(sparsity, Iterable):
        tmp = float(sparsity)
        sparsity = [tmp for _ in range(len(nodes))]

    if len(nodes) != len(sparsity):
        raise ValueError(
            "len(nodes) {} does not match len(sparsity) {}".format(
                len(nodes), len(sparsity)
            )
        )

    for node, sparsity in zip(nodes, sparsity):
        weight, bias = get_node_params(model, node)
        pruned_weight_val = prune_unstructured(weight.val, sparsity)
        update_model_param(model, weight.name, pruned_weight_val)


def prune_model_one_shot_iter(
    model: ModelProto, nodes: List[NodeProto], sparsity: Union[float, List[float]]
):
    """
    Iteratively prune a model in-place with one shot pruning (no retraining) according
    to magnitude pruning. Does so in an unstructured way currently

    :param model: the model to apply pruning to
    :param nodes: the nodes within the model to prune to the desired sparsities
    :param sparsity: the sparsity level to prune all nodes to if a float,
        or the sparsity level to prune each node to if a list of floats
    """
    if not isinstance(sparsity, Iterable):
        tmp = float(sparsity)
        sparsity = [tmp for _ in range(len(nodes))]

    if len(nodes) != len(sparsity):
        raise ValueError(
            "len(nodes) {} does not match len(sparsity) {}".format(
                len(nodes), len(sparsity)
            )
        )

    for index, (node, sparsity) in enumerate(zip(nodes, sparsity)):
        weight, bias = get_node_params(model, node)
        pruned_weight_val = prune_unstructured(weight.val, sparsity)
        update_model_param(model, weight.name, pruned_weight_val)
        yield (index + 1) / len(nodes)
