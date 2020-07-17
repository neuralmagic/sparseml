"""
Code related to applying a mask onto a parameter to impose kernel sparsity,
aka model pruning
"""

from typing import Union, List
from collections import OrderedDict
import numpy
from onnx import numpy_helper, ModelProto
from onnx.helper import make_graph, make_model

from neuralmagicML.onnx.utils import get_node_params


__all__ = ["prune_unstructured", "prune_model_one_shot"]


def prune_unstructured(array: numpy.ndarray, sparsity: float) -> numpy.ndarray:
    """
    Prune a numpy array with unstructured sparsity according to magnitude pruning

    :param array: the array to prune (introduce zeros), will remove the lowest
        absolute values in the array
    :param sparsity: the sparsity value, as a decimal, to impose in the array
    :return: the pruned array
    """
    array = numpy.array(array)  # make a copy because arrays from onnx are read only
    sparse_index = round(sparsity * array.size) - 1

    if sparse_index < 0:
        return array

    sorted_array = numpy.sort(numpy.abs(array.flatten()))
    sparse_thresh = sorted_array[sparse_index]
    array[numpy.abs(array) < sparse_thresh] = 0

    measured_sparsity = float(array.size - numpy.count_nonzero(array)) / float(
        array.size
    )
    assert abs(measured_sparsity - sparsity) < 1e-9

    return array


def prune_model_one_shot(
    model: ModelProto, nodes: List, sparsity: Union[float, List[float]]
) -> ModelProto:
    """
    Prune a model with one shot pruning (no retraining) according to magnitude pruning.
    Does so in an unstructured way currently

    :param model: the model to apply pruning to
    :param nodes: the nodes within the model to prune to the desired sparsities
    :param sparsity: the sparsity level to prune all nodes to if a float,
        or the sparsity level to prune each node to if a list of floats
    :return: the new, pruned model
    """
    if isinstance(sparsity, float):
        tmp = sparsity
        sparsity = [tmp for _ in range(len(nodes))]

    if len(nodes) != len(sparsity):
        raise ValueError(
            "len(nodes) {} does not match len(sparsity) {}".format(
                len(nodes), len(sparsity)
            )
        )

    prune_weights = OrderedDict()

    for node, sparsity in zip(nodes, sparsity):
        weight, bias = get_node_params(model, node)
        prune_weights[weight.name] = (weight.val, sparsity)

    new_inits = []

    for init in model.graph.initializer:
        if init.name in prune_weights:
            weight, sparsity = prune_weights[init.name]
            pruned_weight = prune_unstructured(weight, sparsity)
            init = numpy_helper.from_array(pruned_weight, name=init.name)

        new_inits.append(init)

    pruned_graph = make_graph(
        model.graph.node,
        model.graph.name,
        model.graph.input,
        model.graph.output,
        initializer=new_inits,
        value_info=model.graph.value_info,
    )
    pruned_model = make_model(pruned_graph)

    return pruned_model
