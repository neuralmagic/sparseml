from typing import Union, List, Tuple
import re
import numpy
import tensorflow.contrib.graph_editor as ge
from tensorflow.contrib.graph_editor.util import ListView

from neuralmagicML.tensorflow.utils.helpers import tf_compat

__all__ = [
    "VAR_INDEX_FROM_TRAINABLE",
    "get_op_var_index",
    "clean_tensor_name",
    "get_op_input_var",
    "get_tensor_var",
    "get_prunable_ops",
    "eval_tensor_density",
    "eval_tensor_sparsity",
]


VAR_INDEX_FROM_TRAINABLE = "from_trainable"


def get_op_var_index(var_index: Union[str, int], op_inputs: ListView) -> int:
    """
    Get the index of the variable input to an operation.
    Ex: getting the index of the weight for a convolutional operator.

    | There are a few different modes that this can work as for var_index value:
    |   - int given, treats this as the desired index of the weight
    |   - string given equal to VAR_INDEX_FROM_TRAINABLE, picks the most likely input
    |     based on finding the first trainable variable that is an input.
    |     Defaults to the last input (all convs have input as last and most matmuls)
    |   - string given, attempts to match the string in any of the inputs name.
    |     Uses the first one found, raises an exception if one couldn't be found

    :param var_index: the index to use for figuring out the proper input
    :param op_inputs: inputs to the operator from graph editor
    :return: the integer representing the index of the desired variable
    """
    # check if given index is explicit
    if isinstance(var_index, int):
        if var_index < 0:
            var_index += len(op_inputs)

        return var_index

    # check through trainable vars for best fit
    if var_index == "from_trainable":
        # default to last as this is where tf by default is configured
        # to put the weight variables
        weight_index = len(op_inputs) - 1
        trainable_vars = [var.name for var in tf_compat.trainable_variables()]

        for index, inp in enumerate(op_inputs):
            expected_name = "{}:0".format(clean_tensor_name(inp.name))

            if expected_name in trainable_vars:
                return index

        return weight_index

    # assume that the passed in value is an identifier for the variable name
    for index, inp in enumerate(op_inputs):
        if var_index in inp.name:
            return index

    raise ValueError("unknown value given for var_index of {}".format(var_index))


def clean_tensor_name(var_tens: Union[str, tf_compat.Tensor]) -> str:
    """
    :param var_tens: the tensor to get a variable for
    :return: the cleaned version of the name for a variable tensor
        (removes read and indices at the end)
    """
    name = var_tens if isinstance(var_tens, str) else var_tens.name
    name = re.sub(r"/read:[0-9]+$", "", name)
    name = re.sub(r":[0-9]+$", "", name)

    return name


def get_op_input_var(
    operation: tf_compat.Operation,
    var_index: Union[str, int] = VAR_INDEX_FROM_TRAINABLE,
) -> tf_compat.Tensor:
    """
    Get the input variable for an operation.
    Ex: the weight for a conv operator.
    See @get_op_var_index for proper values for var_index.

    :param operation: the operation to get the input variable for
    :param var_index: the index to guide which input to grab from the operation
    :return: the tensor input that represents the variable input for the operation
    """
    op_sgv = ge.sgv(operation)
    var_index = get_op_var_index(var_index, op_sgv.inputs)

    return op_sgv.inputs[var_index]


def get_tensor_var(tens: tf_compat.Tensor) -> tf_compat.Variable:
    """
    Get the variable associated with a given tensor.
    Raises a ValueError if not found

    :param tens: the tensor to find a variable for
    :return: the found variable matching the given tensor
    """
    expected_name = "{}:0".format(clean_tensor_name(tens))

    for var in tf_compat.trainable_variables():
        if expected_name == var.name:
            return var

    raise ValueError(
        "could not find a trainable variable that matched the tensor {}".format(tens)
    )


def get_prunable_ops(
    graph: tf_compat.Graph = None,
) -> List[Tuple[str, tf_compat.Operation]]:
    """
    Get the prunable operations from a TensorFlow graph.

    :param graph: the graph to get the prunable operations from.
        If not supplied, then will use the default graph
    :return: a list containing the names and ops of the prunable operations
        (MatMul, Conv1D, Conv2D, Conv3D)
    """
    if not graph:
        graph = tf_compat.get_default_graph()

    ops = []

    for op in graph.get_operations():
        if (
            op.type in ["MatMul", "Conv1D", "Conv2D", "Conv3D"]
            and "gradients/" not in op.name
            and "_grad/" not in op.name
        ):
            ops.append((op.name, op))

    return ops


def eval_tensor_density(
    tens: tf_compat.Tensor, sess: tf_compat.Session = None
) -> float:
    """
    Get the density (fraction of non zero values) in a tensor

    :param tens: the tensor to get the density for
    :param sess: the session to use for evaluating the tensor,
        if not supplied will use the default session
    :return: the density of the tensor
    """
    if not sess:
        sess = tf_compat.get_default_session()

    val_array = sess.run(tens)
    num_nonzeros = numpy.count_nonzero(val_array)
    density = float(num_nonzeros) / float(val_array.size)

    return density


def eval_tensor_sparsity(
    tens: tf_compat.Tensor, sess: tf_compat.Session = None
) -> float:
    """
    Get the sparsity (fraction of zero values) in a tensor

    :param tens: the tensor to get the sparsity for
    :param sess: the session to use for evaluating the tensor,
        if not supplied will use the default session
    :return: the sparsity of the tensor
    """
    return 1.0 - eval_tensor_density(tens, sess)
