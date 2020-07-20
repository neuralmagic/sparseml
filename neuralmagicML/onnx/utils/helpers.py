"""
Utility / helper functions
"""

from typing import Tuple, Dict, Union, Any, List, NamedTuple
from collections import OrderedDict
from functools import reduce
import numpy
import onnx
from onnx import numpy_helper, ModelProto
from onnx.helper import get_attribute_value

from neuralmagicML.utils import clean_path


__all__ = [
    "check_load_model",
    "extract_node_id",
    "get_node_by_id",
    "extract_shape",
    "get_init_by_name",
    "NodeParam",
    "conv_node_params",
    "gemm_node_params",
    "matmul_node_params",
    "get_node_params",
    "get_node_attributes",
    "get_node_inputs",
    "get_node_outputs",
    "is_prunable_node",
    "get_prunable_nodes",
    "SparsityMeasurement",
    "onnx_nodes_sparsities",
    "model_inputs",
    "model_outputs",
]


def check_load_model(model: Union[str, ModelProto]) -> ModelProto:
    """
    Load an ONNX model from a given file path if supplied.
    If already a model proto, then returns.

    :param model: the model proto or path to the model onnx file to check for loading
    :return: the loaded onnx ModelProto
    """
    if isinstance(model, ModelProto):
        return model

    if isinstance(model, str):
        return onnx.load(clean_path(model))

    raise ValueError("unknown type given for model: {}".format(model))


def extract_node_id(node) -> str:
    """
    Get the node id for a given node from an onnx model.
    Grabs the first ouput id as the node id.
    This is because is guaranteed to be unique for this node by the onnx spec.

    :param node: the node to grab an id for
    :return: the id for the node
    """
    outputs = node.output

    return str(outputs[0])


def get_node_by_id(model: ModelProto, node_id: str) -> Union[Any, None]:
    """
    Get a node from a model by the node_id generated from extract_node_id

    :param model: the model proto loaded from the onnx file
    :param node_id: id of the node to get from the model
    :return: the retrieved node or None if no node found
    """
    for node in model.graph.node:
        if extract_node_id(node) == node_id:
            return node

    return None


def extract_shape(proto: Any) -> Union[None, Tuple[Union[int, None], ...]]:
    """
    Extract the shape info from a proto.
    Convenient for inputs into a model for example to get the tensor dimension.

    :param proto: the proto to get tensor shape info for
    :return: a tuple containing shape info if found, else None
    """
    tensor_type = proto.type.tensor_type

    if not tensor_type.HasField("shape"):
        return None

    shape = []

    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        else:
            shape.append(None)

    return tuple(shape)


def get_init_by_name(model: ModelProto, init_name: str) -> Union[Any, None]:
    """
    Get an initializer by name from the onnx model proto graph

    :param model: the model proto loaded from the onnx file
    :param init_name: the name of the initializer to retrieve
    :return: the initializer retrieved by name from the model
    """
    matching_inits = [
        init for init in model.graph.initializer if init.name == init_name
    ]

    if len(matching_inits) == 0:
        return None

    if len(matching_inits) > 1:
        raise ValueError(
            "found duplicate inits in the onnx graph for name {} in {}".format(
                init_name, model
            )
        )

    return matching_inits[0]


class NodeParam(NamedTuple):
    """
    Simple named tuple for mapping a node value to the init name it came from
    """

    name: str
    val: numpy.ndarray


def conv_node_params(
    model: ModelProto, node: Any
) -> Tuple[NodeParam, Union[NodeParam, None]]:
    """
    Get the params (weight and bias) for a conv node in an onnx ModelProto

    :param model: the model proto loaded from the onnx file
    :param node: the conv node to get the params for
    :return: a tuple containing the weight, bias (if it is present)
    """
    node_id = extract_node_id(node)

    if str(node.op_type).lower() != "conv":
        raise ValueError("node_id of {} is not a conv: {}".format(node_id, node))

    weight_init = get_init_by_name(model, node.input[1])
    weight = NodeParam(node.input[1], numpy_helper.to_array(weight_init))

    if len(node.input) > 2:
        bias_init = get_init_by_name(model, node.input[2])
        bias = NodeParam(node.input[2], numpy_helper.to_array(bias_init))
    else:
        bias = None

    return weight, bias


def _get_matmul_gemm_weight(model: ModelProto, node: Any) -> NodeParam:
    node_id = extract_node_id(node)

    if str(node.op_type).lower() not in ["gemm", "matmul"]:
        raise ValueError(
            "node id of {} is not a gemm or matmul: {}".format(node_id, node)
        )

    # for gemm, the positions of weights are not explicit in the definition
    weight_inits = [
        get_init_by_name(model, node.input[0]),
        get_init_by_name(model, node.input[1]),
    ]

    # putting this here in case it's changed in the future since the else case below
    # falls to expecting only 2
    assert len(weight_inits) == 2

    if weight_inits[0] is None and weight_inits[1] is None:
        raise ValueError(
            "could not find weight for gemm / matmul node with id {}: {}".format(
                node_id, node
            )
        )
    elif weight_inits[0] is not None and weight_inits[1] is not None:
        raise ValueError(
            "found too many weight inputs for gemm / matmul node with id {}: {}".format(
                node_id, node
            )
        )
    elif weight_inits[0] is not None:
        weight = NodeParam(node.input[0], numpy_helper.to_array(weight_inits[0]))
    else:
        weight = NodeParam(node.input[1], numpy_helper.to_array(weight_inits[1]))

    return weight


def gemm_node_params(
    model: ModelProto, node: Any
) -> Tuple[NodeParam, Union[NodeParam, None]]:
    """
    Get the params (weight and bias) for a gemm node in an onnx ModelProto

    :param model: the model proto loaded from the onnx file
    :param node: the conv node to get the params for
    :return: a tuple containing the weight, bias (if it is present)
    """
    node_id = extract_node_id(node)

    if str(node.op_type).lower() != "gemm":
        raise ValueError("node_id of {} is not a gemm: {}".format(node_id, node))

    weight = _get_matmul_gemm_weight(model, node)

    if len(node.input) > 2:
        bias_init = get_init_by_name(model, node.input[2])
        bias = NodeParam(node.input[2], numpy_helper.to_array(bias_init))
    else:
        bias = None

    return weight, bias


def matmul_node_params(
    model: ModelProto, node: Any
) -> Tuple[NodeParam, Union[NodeParam, None]]:
    """
    Get the params (weight) for a matmul node in an onnx ModelProto.
    In the future will retrieve a following bias addition as the bias for the matmul.

    :param model: the model proto loaded from the onnx file
    :param node: the conv node to get the params for
    :return: a tuple containing the weight, bias (if it is present)
    """
    # todo, expand this to grab a bias add if one occurs after the matmul for fcs
    node_id = extract_node_id(node)

    if str(node.op_type).lower() != "matmul":
        raise ValueError("node_id of {} is not a matmul: {}".format(node_id, node))

    weight = _get_matmul_gemm_weight(model, node)
    bias = None

    return weight, bias


def get_node_params(
    model: ModelProto, node: Any
) -> Tuple[NodeParam, Union[NodeParam, None]]:
    """
    Get the params (weight and bias) for a node in an onnx ModelProto.
    Must be an op type of one of [conv, gemm, matmul]

    :param model: the model proto loaded from the onnx file
    :param node: the conv node to get the params for
    :return: a tuple containing the weight, bias (if it is present)
    """
    node_id = extract_node_id(node)

    if str(node.op_type).lower() == "conv":
        return conv_node_params(model, node)

    if str(node.op_type).lower() == "gemm":
        return gemm_node_params(model, node)

    if str(node.op_type).lower() == "matmul":
        return matmul_node_params(model, node)

    raise ValueError(
        (
            "node_id of {} is not a supported node (conv, gemm, matmul) "
            "for params: {}"
        ).format(node_id, node)
    )


def get_node_attributes(node: Any) -> Dict[str, Any]:
    """
    :param node: the ONNX node to get the attibutes for
    :return: a dictionary containing all attributes for the node
    """
    attributes = reduce(
        lambda accum, attribute: accum.update(
            {attribute.name: get_attribute_value(attribute)}
        )
        or accum,
        node.attribute,
        {},
    )

    for key in list(attributes.keys()):
        val = attributes[key]

        if not (
            isinstance(val, int)
            or isinstance(val, float)
            or isinstance(val, str)
            or isinstance(val, list)
            or isinstance(val, dict)
        ):
            attributes[key] = None

    return attributes


def get_node_inputs(model: ModelProto, node: Any) -> List[str]:
    """
    :param model: the model the node is from
    :param node: the node to get all inputs (non initializers) for
    :return: the names off all the inputs to the node that are not initializers
    """
    inputs = []

    for inp in node.input:
        if get_init_by_name(model, inp) is None:
            inputs.append(inp)

    return inputs


def get_node_outputs(model: ModelProto, node: Any) -> List[str]:
    """
    :param model: the model the node is from
    :param node: the node to get all outputs (non initializers) for
    :return: the names of all the outputs to the node that are not initializers
    """
    outputs = []

    for out in node.output:
        if get_init_by_name(model, out) is None:
            outputs.append(out)

    return outputs


def is_prunable_node(node: Any) -> bool:
    """
    :param node: an onnx node or op_type string
    :return: True if the given node or op type is prunable, False othewise
    """
    prunable_types = ["conv", "gemm", "matmul"]

    return (
        str(node.op_type if not isinstance(node, str) else node).lower()
        in prunable_types
    )


def get_prunable_nodes(model: Union[str, ModelProto]) -> List[Any]:
    """
    Get the prunable nodes in an onnx model proto.
    Prunable nodes are defined as any conv, gemm, or matmul

    :param model: the model proto loaded from the onnx file
    :return: a list of nodes from the model proto
    """
    model = check_load_model(model)
    prunable_types = ["conv", "gemm", "matmul"]
    prunable_nodes = []

    for node in model.graph.node:
        if str(node.op_type).lower() in prunable_types:
            prunable_nodes.append(node)

    return prunable_nodes


class SparsityMeasurement(NamedTuple):
    """
    A measurement of the sparsity for a given onnx node or model
    """

    node_id: str
    params_count: int
    params_zero_count: int
    sparsity: float
    density: float


def onnx_nodes_sparsities(
    model: Union[str, ModelProto],
) -> Tuple[SparsityMeasurement, Dict[str, SparsityMeasurement]]:
    """
    Retrieve the sparsities for each Conv or Gemm op in an onnx graph
    for the associated weight inputs.

    :param model: onnx model to use
    :return: a tuple containing the overall sparsity measurement for the model,
        each conv or gemm node found in the model
    """
    model = check_load_model(model)
    node_inp_sparsities = OrderedDict()  # type: Dict[str, SparsityMeasurement]
    params_count = 0
    params_zero_count = 0

    for node in get_prunable_nodes(model):
        node_id = extract_node_id(node)
        node_key = "{}(id={})".format(node.op_type, node_id)
        weight, bias = get_node_params(model, node)

        zeros = weight.val.size - numpy.count_nonzero(weight.val)
        sparsity = float(zeros) / float(weight.val.size)
        density = 1.0 - sparsity
        node_inp_sparsities[
            "{}_inp={}".format(node_key, weight.name)
        ] = SparsityMeasurement(node_id, weight.val.size, zeros, sparsity, density)

        params_count += weight.val.size
        params_zero_count += zeros

    return (
        SparsityMeasurement(
            "ModelProto",
            params_count,
            params_zero_count,
            float(params_zero_count) / float(params_count),
            float(params_count - params_zero_count) / float(params_count),
        ),
        node_inp_sparsities,
    )


def model_inputs(model: Union[str, ModelProto]) -> List:
    """
    Get the input to the model from an ONNX model

    :param model: the loaded model or a file path to the onnx model
        to get the model inputs for
    :return: the input to the model
    """
    model = check_load_model(model)
    inputs_all = [node.name for node in model.graph.input]
    inputs_init = [node.name for node in model.graph.initializer]
    input_names = list(set(inputs_all) - set(inputs_init))
    inputs = [node for node in model.graph.input if node.name in input_names]
    assert len(input_names) == len(inputs)

    return inputs


def model_outputs(model: Union[str, ModelProto]) -> List:
    """
    Get the output from an ONNX model

    :param model: the loaded model or a file path to the onnx model
        to get the model outputs for
    :return: the output from the model
    """
    model = check_load_model(model)
    outputs = [node for node in model.graph.output]

    return outputs
