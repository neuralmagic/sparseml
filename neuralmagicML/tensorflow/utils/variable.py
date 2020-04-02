from typing import Union
import re
import tensorflow.contrib.graph_editor as ge
from tensorflow.contrib.graph_editor.util import ListView

from neuralmagicML.tensorflow.utils.helpers import tf_compat

__all__ = [
    "VAR_INDEX_FROM_TRAINABLE",
    "get_op_var_index",
    "get_var_name",
    "get_op_input_var",
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
            expected_name = re.sub(r"/read:[0-9]+$", "", inp.name)

            if expected_name in trainable_vars:
                return index

        return weight_index

    # assume that the passed in value is an identifier for the variable name
    for index, inp in enumerate(op_inputs):
        if var_index in inp.name:
            return index

    raise ValueError("unknown value given for var_index of {}".format(var_index))


def get_var_name(var_tens: tf_compat.Tensor) -> str:
    """
    :param var_tens: the tensor to get a variable for
    :return: the cleaned version of the name for a variable tensor
        (removes read at the end)
    """
    # remove the 'read' name, if present in the variable
    return re.sub(r"/read:[0-9]+$", "", var_tens.name)


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
