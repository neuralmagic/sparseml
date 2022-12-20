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
Utility / helper functions
"""

import logging
from collections import OrderedDict
from copy import deepcopy
from functools import reduce
from typing import Any, Dict, List, NamedTuple, Tuple, Union

import numpy
import onnx
from onnx import ModelProto, NodeProto, TensorProto, numpy_helper
from onnx.helper import get_attribute_value, make_empty_tensor_value_info

from sparseml.onnx.base import require_onnxruntime
from sparseml.utils import clean_path


_LOGGER = logging.getLogger(__name__)

__all__ = [
    "validate_onnx_file",
    "check_load_model",
    "extract_node_id",
    "get_node_by_id",
    "get_nodes_by_input_id",
    "get_nodes_by_output_id",
    "extract_shape",
    "get_numpy_dtype",
    "NodeShape",
    "extract_nodes_shapes_ort",
    "extract_nodes_shapes_shape_inference",
    "extract_node_shapes",
    "get_init_by_name",
    "get_attr_float_val_for_node",
    "NodeParam",
    "conv_node_params",
    "gemm_node_params",
    "matmul_node_params",
    "get_node_params",
    "BatchNormParams",
    "get_batch_norm_params",
    "get_node_attributes",
    "get_node_inputs",
    "get_node_outputs",
    "get_node_input_nodes",
    "get_node_output_nodes",
    "is_prunable_node",
    "is_foldable_node",
    "get_prunable_node_from_foldable",
    "get_prunable_nodes",
    "SparsityMeasurement",
    "onnx_nodes_sparsities",
    "model_inputs",
    "model_outputs",
    "get_kernel_shape",
    "calculate_flops",
    "get_quantize_parent_for_dequantize_node",
    "get_tensor_shape",
    "get_tensor_dim_shape",
    "set_tensor_dim_shape",
]


def validate_onnx_file(path: str):
    """
    Validate that a file at a given path is a valid ONNX model

    :param path: the path of the file to validate
    :raise ValueError: if not a valid ONNX model
    """
    try:
        onnx_model = check_load_model(path)
        onnx.checker.check_model(onnx_model)
        if not onnx_model.opset_import:
            raise ValueError("could not parse opset_import")
    except Exception as err:
        raise ValueError(f"Invalid onnx model: {err}")


def check_load_model(model: Union[str, ModelProto]) -> ModelProto:
    """
    Load an ONNX model from a given file path if supplied.
    If already a model proto, then returns.

    :param model: the model proto or path to the model ONNX file to check for loading
    :return: the loaded ONNX ModelProto
    """
    if isinstance(model, ModelProto):
        return model

    if isinstance(model, str):
        return onnx.load(clean_path(model))

    raise ValueError(f"unknown type given for model: {type(model)}")


def extract_node_id(node: NodeProto) -> str:
    """
    Get the node id for a given node from an ONNX model.
    Grabs the first ouput id as the node id.
    This is because is guaranteed to be unique for this node by the ONNX spec.

    :param node: the node to grab an id for
    :return: the id for the node
    """
    outputs = node.output

    return str(outputs[0])


def get_node_by_id(model: ModelProto, node_id: str) -> Union[NodeProto, None]:
    """
    Get a node from a model by the node_id generated from extract_node_id

    :param model: the model proto loaded from the ONNX file
    :param node_id: id of the node to get from the model
    :return: the retrieved node or None if no node found
    """
    for node in model.graph.node:
        if extract_node_id(node) == node_id:
            return node

    return None


def get_nodes_by_input_id(model: ModelProto, input_id: str) -> List[NodeProto]:
    """
    Get all the nodes in a model that have a given id as one of the inputs

    :param model: the model proto loaded from the ONNX file
    :param input_id: id of the input to get nodes by
    :return: the retrieved nodes
    """
    nodes = []

    for node in model.graph.node:
        inputs = get_node_inputs(model, node)

        if input_id in inputs:
            nodes.append(node)

    return nodes


def get_nodes_by_output_id(model: ModelProto, output_id: str) -> List[NodeProto]:
    """
    Get all the nodes in a model that have a given id as one of the outputs

    :param model: the model proto loaded from the ONNX file
    :param output_id: id of the output to get nodes by
    :return: the retrieved nodes
    """
    nodes = []

    for node in model.graph.node:
        outputs = get_node_outputs(model, node)

        if output_id in outputs:
            nodes.append(node)

    return nodes


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


def get_numpy_dtype(tensor: onnx.TensorProto) -> Union[None, numpy.dtype]:
    """
    Extract the NumPy dtype of an ONNX tensor.
    Returns None if there is not a direct mapping from the ONNX data type
    to a NumPy dtype.

    :param tensor: the tensor to get the dtype of
    :return: a NumPy dtype for the tensor if available otherwise None
    """
    data_type_enum = tensor.type.tensor_type.elem_type  # type: int
    data_type = onnx.TensorProto.DataType.Name(data_type_enum).lower()  # type: str
    if data_type == "float":
        data_type = "float32"

    if hasattr(numpy, data_type):
        return getattr(numpy, data_type)
    return None


"""
Tuple containing a node id and its input and output shapes
"""
NodeShape = NamedTuple(
    "NodeShape",
    [
        ("id", str),
        ("input_shapes", Union[List[List[int]], None]),
        ("output_shapes", Union[List[List[int]], None]),
    ],
)


@require_onnxruntime()
def extract_nodes_shapes_ort(model: ModelProto) -> Dict[str, List[List[int]]]:
    """
    Creates a modified model to expose intermediate outputs and runs an ONNX Runtime
    InferenceSession to obtain the output shape of each node.

    :param model: an ONNX model
    :return: a list of NodeArg with their shape exposed
    """
    import onnxruntime  # import protected by @require_onnxruntime()

    model_copy = deepcopy(model)

    for node in model_copy.graph.node:
        intermediate_layer_value_info = make_empty_tensor_value_info(
            extract_node_id(node)
        )
        model_copy.graph.output.append(intermediate_layer_value_info)

    sess_options = onnxruntime.SessionOptions()
    sess_options.log_severity_level = 3
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if onnxruntime.get_device() == "GPU"
        else ["CPUExecutionProvider"]
    )

    sess = onnxruntime.InferenceSession(
        model_copy.SerializeToString(),
        sess_options=sess_options,
        providers=providers,
    )

    output_shapes = {}
    for node in sess.get_outputs() + sess.get_inputs():
        output_shapes[node.name] = (
            node.shape if node.shape is not None and len(node.shape) > 0 else None
        )
    return output_shapes


def extract_nodes_shapes_shape_inference(
    model: ModelProto,
) -> Dict[str, List[Union[None, List[int]]]]:
    """
    Creates a modified model to expose intermediate outputs and runs an ONNX shape
    inference to obtain the output shape of each node.

    NOTE: The ONNX docs on shape inference have the following
    disclaimer on shape inference:
    Shape inference is not guaranteed to be complete.
    In particular, some dynamic behaviors block the flow of shape inference,
    for example a Reshape to a dynamically-provide shape.
    Also, all operators are not required to have a shape inference implementation.

    :param model: an ONNX model
    :return: a list of NodeProto with their shape exposed
    """
    model_copy = deepcopy(model)

    for node in model_copy.graph.node:
        model_copy.graph.output.extend(
            [
                onnx.helper.make_tensor_value_info(
                    output, onnx.TensorProto.UNDEFINED, None
                )
                for output in node.output
            ]
        )

    if hasattr(onnx, "shape_inference"):
        model_copy = onnx.shape_inference.infer_shapes(model_copy)
    else:
        raise ModuleNotFoundError(
            "onnx.shape_inference not available for current version, "
            "please upgrade to use this functionality"
        )

    output_shapes = {}
    for node in model_copy.graph.output:
        node_shape = extract_shape(node)
        output_shapes[node.name] = (
            list(node_shape) if node_shape is not None and len(node_shape) > 0 else None
        )

    return output_shapes


def extract_node_shapes(model: ModelProto) -> Dict[str, NodeShape]:
    """
    Extracts the shape information for each node as a NodeShape object.

    :param model: the loaded onnx.ModelProto to extract node shape information from
    :return: a mapping of node id to a NodeShape object
    """

    # Maps NodeArg to its inputs
    node_to_inputs = {}
    for node in model.graph.node:
        node_to_inputs[extract_node_id(node)] = node.input

    # Obtains output shapes for each model's node
    output_shapes = None

    try:
        output_shapes = extract_nodes_shapes_ort(model)
    except Exception as err:
        _LOGGER.warning(
            "Extracting shapes using ONNX Runtime session failed: {}".format(err)
        )

    if output_shapes is None:
        try:
            output_shapes = extract_nodes_shapes_shape_inference(model)
        except Exception as err:
            _LOGGER.warning(
                "Extracting shapes using ONNX shape_inference failed: {}".format(err)
            )

    # Obtains the input shapes for each node
    if output_shapes is None:
        output_shapes = {}

    input_shapes = {}

    for node in output_shapes.keys():
        if node not in node_to_inputs:
            continue
        input_shapes[node] = [
            output_shapes[input_node]
            for input_node in node_to_inputs[node]
            if input_node in output_shapes and output_shapes[input_node] is not None
        ]
        input_shapes[node] = input_shapes[node] if len(input_shapes[node]) > 0 else None

    # Combines shape information into mapping of node id to a NodeShape object
    node_shapes = {}
    for node in output_shapes.keys():
        node_shapes[node] = NodeShape(
            node,
            input_shapes[node] if node in input_shapes else None,
            [output_shapes[node]]
            if node in output_shapes and output_shapes[node] is not None
            else None,
        )

    def _fix_shapes(shapes: List[Union[List[Union[int, None, str]], None]]):
        if not shapes:
            return

        for shape in shapes:
            if not shape:
                continue

            for index, index_shape in enumerate(shape):
                try:
                    shape[index] = (
                        round(index_shape)
                        if isinstance(index_shape, float)
                        else int(index_shape)
                    )
                except Exception:
                    # not parsable as an int (none or string)
                    # set to None
                    shape[index] = None

    for node_id, node_shape in node_shapes.items():
        _fix_shapes(node_shape.input_shapes)
        _fix_shapes(node_shape.output_shapes)

    return node_shapes


def get_init_by_name(model: ModelProto, init_name: str) -> Union[Any, None]:
    """
    Get an initializer by name from the ONNX model proto graph

    :param model: the model proto loaded from the ONNX file
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
            "found duplicate inits in the ONNX graph for name {} in {}".format(
                init_name, model
            )
        )

    return matching_inits[0]


def _get_node_param_init_by_idx(
    model: ModelProto, node: onnx.NodeProto, idx: int
) -> Union[numpy.ndarray, None]:
    if idx < len(node.input):
        initializer = get_init_by_name(model, node.input[idx])
        if initializer is not None:
            return numpy_helper.to_array(initializer)
    return None


def get_attr_float_val_for_node(node: onnx.NodeProto, attr: str) -> Union[float, None]:
    """
    :param node: Node to get the attribute value of
    :param attr: Attribute name to match in the node
    :return: The value of the attribute if the attribute found in the node and is
        a float type. Otherwise returns None
    """
    attr = attr.lower()
    node_attr_matches = [att for att in node.attribute if attr in att.name]
    if not node_attr_matches:
        return None
    node_attr = node_attr_matches[0]
    return node_attr.f if node_attr.type == node_attr.FLOAT else None


"""
Simple named tuple for mapping a node value to the init name it came from
"""
NodeParam = NamedTuple("NodeParam", [("name", str), ("val", numpy.ndarray)])

_TRIVIAL_OP_TYPES = {"Reshape", "Transpose"}


def _get_init_by_name_nested(
    model, weight_name
) -> Tuple[Union[str, None], Union[TensorProto, None]]:
    # traverses graph if weights are reshaped / transposed before becoming layer input
    init = get_init_by_name(model, weight_name)

    if init is not None:
        return weight_name, init

    nested = None, None
    parent_nodes = get_nodes_by_output_id(model, weight_name)

    if not parent_nodes:
        return nested

    for parent in parent_nodes:
        if parent.op_type not in _TRIVIAL_OP_TYPES:
            continue

        nested = _get_init_by_name_nested(model, parent.input[0])

        if nested[0] is not None and nested[1] is not None:
            break

    return nested


def conv_node_params(
    model: ModelProto, node: NodeProto, include_values: bool = True
) -> Tuple[NodeParam, Union[NodeParam, None]]:
    """
    Get the params (weight and bias) for a conv node in an ONNX ModelProto

    :param model: the model proto loaded from the ONNX file
    :param node: the conv node to get the params for
    :param include_values: True to include the param values as NumPy arrays
        in the returned NodeParam objects.
        False to not load the values -- in this event NodeParam.val will be None
    :return: a tuple containing the weight, bias (if it is present)
    """
    node_id = extract_node_id(node)

    if str(node.op_type).lower() != "conv":
        raise ValueError("node_id of {} is not a conv: {}".format(node_id, node))

    weight_name, weight_init = _get_init_by_name_nested(model, node.input[1])
    weight = NodeParam(
        weight_name, numpy_helper.to_array(weight_init) if include_values else None
    )
    if len(node.input) > 2:
        bias_name, bias_init = _get_init_by_name_nested(model, node.input[2])
        bias = NodeParam(
            bias_name, numpy_helper.to_array(bias_init) if include_values else None
        )
    else:
        bias = None

    return weight, bias


def _get_matmul_gemm_weight(
    model: ModelProto, node: NodeProto, include_values: bool = True
) -> NodeParam:
    node_id = extract_node_id(node)

    if str(node.op_type).lower() not in ["gemm", "matmul"]:
        raise ValueError(
            "node id of {} is not a gemm or matmul: {}".format(node_id, node)
        )

    # for gemm, the positions of weights are not explicit in the definition
    weight_inits = [
        _get_init_by_name_nested(model, node.input[0]),
        _get_init_by_name_nested(model, node.input[1]),
    ]

    # putting this here in case it's changed in the future since the else case below
    # falls to expecting only 2
    assert len(weight_inits) == 2

    if weight_inits[0][1] is None and weight_inits[1][1] is None:
        raise ValueError(
            "could not find weight for gemm / matmul node with id {}: {}".format(
                node_id, node
            )
        )
    elif weight_inits[0][1] is not None and weight_inits[1][1] is not None:
        raise ValueError(
            "found too many weight inputs for gemm / matmul node with id {}: {}".format(
                node_id, node
            )
        )
    elif weight_inits[0][1] is not None:
        weight = NodeParam(
            weight_inits[0][0],
            numpy_helper.to_array(weight_inits[0][1]) if include_values else None,
        )
    else:
        weight = NodeParam(
            weight_inits[1][0],
            numpy_helper.to_array(weight_inits[1][1]) if include_values else None,
        )

    return weight


def gemm_node_params(
    model: ModelProto, node: NodeProto, include_values: bool = True
) -> Tuple[NodeParam, Union[NodeParam, None]]:
    """
    Get the params (weight and bias) for a gemm node in an ONNX ModelProto

    :param model: the model proto loaded from the ONNX file
    :param node: the conv node to get the params for
    :param include_values: True to include the param values as NumPy arrays
        in the returned NodeParam objects.
        False to not load the values -- in this event NodeParam.val will be None
    :return: a tuple containing the weight, bias (if it is present)
    """
    node_id = extract_node_id(node)

    if str(node.op_type).lower() != "gemm":
        raise ValueError("node_id of {} is not a gemm: {}".format(node_id, node))

    weight = _get_matmul_gemm_weight(model, node, include_values)

    if len(node.input) > 2:
        bias_name, bias_init = _get_init_by_name_nested(model, node.input[2])
        bias = NodeParam(
            bias_name, numpy_helper.to_array(bias_init) if include_values else None
        )
    else:
        bias = None

    return weight, bias


def matmul_node_params(
    model: ModelProto, node: NodeProto, include_values: bool = True
) -> Tuple[NodeParam, Union[NodeParam, None]]:
    """
    Get the params (weight) for a matmul node in an ONNX ModelProto.
    In the future will retrieve a following bias addition as the bias for the matmul.

    :param model: the model proto loaded from the ONNX file
    :param node: the conv node to get the params for
    :param include_values: True to include the param values as NumPy arrays
        in the returned NodeParam objects.
        False to not load the values -- in this event NodeParam.val will be None
    :return: a tuple containing the weight, bias (if it is present)
    """
    # todo, expand this to grab a bias add if one occurs after the matmul for fcs
    node_id = extract_node_id(node)

    if str(node.op_type).lower() != "matmul":
        raise ValueError("node_id of {} is not a matmul: {}".format(node_id, node))

    weight = _get_matmul_gemm_weight(model, node, include_values)
    bias = None

    return weight, bias


def get_node_params(
    model: ModelProto, node: NodeProto, include_values: bool = True
) -> Tuple[NodeParam, Union[NodeParam, None]]:
    """
    Get the params (weight and bias) for a node in an ONNX ModelProto.
    Must be an op type of one of [conv, gemm, matmul]

    :param model: the model proto loaded from the ONNX file
    :param node: the conv node to get the params for
    :param include_values: True to include the param values as NumPy arrays
        in the returned NodeParam objects.
        False to not load the values -- in this event NodeParam.val will be None
    :return: a tuple containing the weight, bias (if it is present)
    """
    node_id = extract_node_id(node)

    if str(node.op_type).lower() == "conv":
        return conv_node_params(model, node, include_values)

    if str(node.op_type).lower() == "gemm":
        return gemm_node_params(model, node, include_values)

    if str(node.op_type).lower() == "matmul":
        return matmul_node_params(model, node, include_values)

    raise ValueError(
        (
            "node_id of {} is not a supported node (conv, gemm, matmul) "
            "for params: {}"
        ).format(node_id, node)
    )


"""
Named tuple for defining the paramters of a batch normalization operator
"""
BatchNormParams = NamedTuple(
    "BatchNormParams",
    [
        ("epsilon", float),
        ("momentum", numpy.ndarray),
        ("scale", numpy.ndarray),
        ("bias", numpy.ndarray),
        ("mean", numpy.ndarray),
        ("var", numpy.ndarray),
    ],
)


def get_batch_norm_params(
    model: onnx.ModelProto, bn_node: onnx.NodeProto
) -> BatchNormParams:
    """
    Get the params and relevant attributes of a batch normalization operator.
    Following the ONNX operators spec, will default epsilon and momentum to
    1e-5 and 0.9 respectively when not defined.

    :param model: the model proto loaded from the ONNX file
    :param bn_node: the batch normalization node to get the params for
    :return: a BatchNormParams named tuple
    """
    bn_attributes = get_node_attributes(bn_node)
    return BatchNormParams(
        epsilon=(bn_attributes["epsilon"] if "epsilon" in bn_attributes else 1e-5),
        momentum=(bn_attributes["momentum"] if "momentum" in bn_attributes else 0.9),
        scale=_get_node_param_init_by_idx(model, bn_node, 1),
        bias=_get_node_param_init_by_idx(model, bn_node, 2),
        mean=_get_node_param_init_by_idx(model, bn_node, 3),
        var=_get_node_param_init_by_idx(model, bn_node, 4),
    )


def get_node_attributes(node: NodeProto) -> Dict[str, Any]:
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


def get_node_inputs(model: ModelProto, node: NodeProto) -> List[str]:
    """
    :param model: the model the node is from
    :param node: the node to get all inputs (non initializers) for
    :return: the names of all the inputs to the node that are not initializers
    """
    inputs = []

    for inp in node.input:
        if get_init_by_name(model, inp) is None:
            inputs.append(inp)

    return inputs


def get_node_outputs(model: ModelProto, node: NodeProto) -> List[str]:
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


def get_node_input_nodes(model: ModelProto, node: NodeProto) -> List[NodeProto]:
    """
    Get all of the nodes that share an output edge for the inputs to a given node

    :param model: the model the node is from
    :param node: the node to get all input nodes for
    :return: the list of nodes that share an output edge
        for the inputs to the given node
    """
    nodes = []
    inputs = [node.name for node in model.graph.input] + [
        node.name for node in model.graph.initializer
    ]

    for input_id in get_node_inputs(model, node):
        if input_id in inputs:
            continue

        nodes.extend(get_nodes_by_output_id(model, input_id))

    return nodes


def get_node_output_nodes(model: ModelProto, node: NodeProto) -> List[NodeProto]:
    """
    Get all of the nodes that share an input edge for the outputs from a given node

    :param model: the model the node is from
    :param node: the node to get all output nodes for
    :return: the list of nodes that share an input edge
        for the outputs from the given node
    """
    nodes = []

    for output_id in get_node_outputs(model, node):
        nodes.extend(get_nodes_by_input_id(model, output_id))

    return nodes


def is_prunable_node(model: ModelProto, node: NodeProto) -> bool:
    """
    :param model: the model the node is from
    :param node: an ONNX node or op_type string
    :return: True if the given node is prunable, False otherwise
    """
    prunable_types = ["conv", "gemm", "matmul"]

    if str(node.op_type).lower() not in prunable_types:
        return False

    try:
        # try to get the weight param, if this fails then
        # it's not a trainable version of the node and therefore not prunable
        get_node_params(model, node, include_values=False)
    except Exception:
        return False

    return True


def is_foldable_node(node: Union[str, NodeProto]) -> bool:
    """
    Foldable nodes as defined by ONNX Runtime and what it supports layerwise folding
    in the ONNX graphs. More info can be found in their docs:
    https://www.onnxruntime.ai/docs/resources/graph-optimizations.html

    :param node: the node or node type to check if it is foldable or not
        according to the ONNX Runtime specs
    :return: True if the node is foldable and therefore can be combined with other
        nodes, False otherwise
    """

    return (node.lower() if isinstance(node, str) else str(node.op_type).lower()) in [
        "batchnormalization",
        "add",
        "mul",
    ]


def get_prunable_node_from_foldable(
    model: ModelProto,
    foldable_node: Union[str, NodeProto],
    traverse_previous: bool = True,
    max_node_distance: int = 3,
) -> Union[None, NodeProto]:
    """
    Get a prunable node that is attached by foldable nodes to a given foldable node.
    Returns None if nothing could be found.
    Ex: get the convolution that would be folded for an attached BatchNormalization

    :param model: the model the node is from
    :param foldable_node: the foldable node or node id to find prunable node from
    :param traverse_previous: True to only search for previous prunable nodes that the
        foldable node could have been attached to for Conv -> BN patterns.
        False to only search for following prunable nodes that the foldable node
        could have been attached to for BN -> Conv patterns.
    :param max_node_distance: The maximum distance
        (and therefore number of foldable nodes) the prunable node must be within
        to match. Ex: max_node_distance = 3, the prunable node must be within 3
        other foldable nodes of the foldable node passed in to match
    :return: the found prunable node
    """
    if isinstance(foldable_node, str):
        foldable_node = get_node_by_id(model, foldable_node)

    if not is_foldable_node(foldable_node):
        raise ValueError(
            "non foldable node passed in for foldable_node: {}".format(
                extract_node_id(foldable_node)
            )
        )

    prunable_node = foldable_node
    num_steps = 0

    while (
        prunable_node is not None
        and not is_prunable_node(model, prunable_node)
        and is_foldable_node(prunable_node)
        and num_steps < max_node_distance
    ):
        next_nodes = (
            get_node_input_nodes(model, prunable_node)
            if traverse_previous
            else get_node_output_nodes(model, prunable_node)
        )
        num_steps += 1
        prunable_node = next_nodes[0] if next_nodes else None

    return (
        None
        if prunable_node is None or not is_prunable_node(model, prunable_node)
        else prunable_node
    )


def get_prunable_nodes(model: Union[str, ModelProto]) -> List[Any]:
    """
    Get the prunable nodes in an ONNX model proto.
    Prunable nodes are defined as any conv, gemm, or matmul

    :param model: the model proto loaded from the ONNX file
    :return: a list of nodes from the model proto
    """
    model = check_load_model(model)
    prunable_nodes = []

    for node in model.graph.node:
        if is_prunable_node(model, node):
            prunable_nodes.append(node)

    return prunable_nodes


"""
A measurement of the sparsity for a given ONNX node or model
"""
SparsityMeasurement = NamedTuple(
    "SparsityMeasurement",
    [
        ("node_id", str),
        ("params_count", int),
        ("params_zero_count", int),
        ("sparsity", float),
        ("density", float),
    ],
)


def onnx_nodes_sparsities(
    model: Union[str, ModelProto],
) -> Tuple[SparsityMeasurement, Dict[str, SparsityMeasurement]]:
    """
    Retrieve the sparsities for each Conv or Gemm op in an ONNX graph
    for the associated weight inputs.

    :param model: ONNX model to use
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

    :param model: the loaded model or a file path to the ONNX model
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

    :param model: the loaded model or a file path to the ONNX model
        to get the model outputs for
    :return: the output from the model
    """
    model = check_load_model(model)
    outputs = [node for node in model.graph.output]

    return outputs


def get_kernel_shape(attributes: Dict[str, Any]) -> Union[List[float], None]:
    """
    Get the kernel shape from a dictionary of a model's attributes

    :param attributes: a dictionary of a model's attributes
    :return: the kernel shape if attribute contains either the kernel or
        kernel_shape field, otherwise None
    """
    if "kernel" in attributes:
        return attributes["kernel"]
    elif "kernel_shape" in attributes:
        return attributes["kernel_shape"]
    else:
        return None


def calculate_flops(
    op_type: str,
    input_shape: Union[List[List], None] = None,
    output_shape: Union[List[List], None] = None,
    weight_shape: Union[List, None] = None,
    kernel_shape: Union[List, None] = None,
    bias_shape: Union[List, None] = None,
    attributes: Union[None, Dict[str, Any]] = None,
) -> Union[float, None]:
    """
    Calculate flops based on operation type and shape of certain attributes.
    If any fields necessary in operation are set to None, will return None

    :param op_type: Operation type of flop calculation
    :param input_shape: List of input shapes of operation
    :param output_shape: List of output shapes of operation
    :param weight_shape: Shape of weights in operation if any, else None
    :param kernel_shape: Shape of kernel in operation if any, else None
    :param bias_shape: Shape of bias in operation if any, else None
    :param attributes: The node attributes if any, else None
    :return: The amount of floating point operations in the operation
    """
    input_shape = _array_as_numeric(input_shape)
    output_shape = _array_as_numeric(output_shape)
    weight_shape = _array_as_numeric(weight_shape)
    kernel_shape = _array_as_numeric(kernel_shape)
    bias_shape = _array_as_numeric(bias_shape)

    if (
        op_type == "Add"
        or op_type == "Mul"
        or op_type == "Div"
        or op_type == "Sub"
        or op_type == "Clip"
    ):
        flops = _numpy_prod_with_none_check(output_shape)
    elif (
        op_type == "Relu"
        or op_type == "LeakyRelu"
        or op_type == "Sigmoid"
        or op_type == "Tanh"
        or op_type == "BatchNormalization"
    ):
        flops = _numpy_prod_with_none_check(output_shape)
    elif op_type == "GlobalAveragePool" or op_type == "GlobalMaxPool":
        flops = _numpy_prod_with_none_check(input_shape)
    elif op_type == "MaxPool" or op_type == "AveragePool":
        flops = (
            numpy.prod(output_shape) * numpy.prod(kernel_shape)
            if output_shape is not None and kernel_shape is not None
            else None
        )
    elif op_type == "MatMul":
        flops = _calculate_flops_matmul(
            op_type,
            input_shape=input_shape,
            output_shape=output_shape,
            weight_shape=weight_shape,
        )
    elif op_type == "Gemm":
        flops = _numpy_prod_with_none_check(weight_shape)
        flops = flops * 2 if flops is not None else None
    elif op_type == "Conv":
        flops = (
            numpy.prod(kernel_shape) * weight_shape[1] * numpy.prod(output_shape)
            if kernel_shape is not None
            and weight_shape is not None
            and output_shape is not None
            else None
        )

        if (
            flops
            and attributes
            and "group" in attributes
            and attributes["group"]
            and attributes["group"] > 1
        ):
            # adjust flops for group / depthwise convolutions
            flops = flops / attributes["group"]
    else:
        flops = None

    if flops is not None and bias_shape is not None:
        if op_type == "Conv":
            flops += numpy.prod(bias_shape) * output_shape[0][-1] * output_shape[0][-2]
        else:
            flops += numpy.prod(bias_shape)

    return flops


def _calculate_flops_matmul(
    op_type: str,
    input_shape: Union[List[List], None] = None,
    output_shape: Union[List[List], None] = None,
    weight_shape: Union[List, None] = None,
) -> Union[float, None]:
    """
    Calculates flops in an ONNX MatMul operation.

    If input shape only contains 1 input, in otherwords the value of the
    first index is 1, then the matrix operation is treated as a Gemm operation.

    Otherwise the operation is treated like a NumPy operation.

    Will return none if any required value is set to None

    :param op_type: Operation type of flop calculation
    :param input_shape: List of input shapes of operation
    :param output_shape: List of output shapes of operation
    :param weight_shape: Shape of weights in operation if any, else None
    :return: The amount of floating point operations in the operation
    """
    flops = None
    if (
        input_shape is not None
        and output_shape is not None
        and len(input_shape) > 1
        and input_shape[0][-1] == input_shape[1][-2]
    ):
        matrix_ops = (
            input_shape[0][-2] * input_shape[1][-1] * (2 * input_shape[0][-1] - 1)
        )
        flops = numpy.prod(output_shape[0][:-2]) * matrix_ops
    elif input_shape is not None and len(input_shape) == 1:
        flops = _numpy_prod_with_none_check(weight_shape)
        flops = flops * 2 if flops is not None else None
    return flops


def _numpy_prod_with_none_check(array: Union[List, None]) -> Union[float, None]:
    """
    :param array: an array like list
    :return: the product of the array if array is not None otherwise return None
    """
    return numpy.prod(array) if array is not None else None


def _attempt_cast_as_float(value: Any) -> float:
    """
    :param vale: a value
    :return: the value as a float if casting is possible, otherwise return 1
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return 1.0


def _array_as_numeric(array: Union[List, None]) -> Union[List, None]:
    """
    :param array: an array like list
    :return: the array with any non numeric or None values replaced with 1
        if array itself is not None, otherwise return None
    """
    if array is None:
        return None

    array = numpy.array(array, dtype=object)
    # Check if the array datatype is a number
    if numpy.issubdtype(array.dtype, numpy.number):
        return array
    else:
        to_float = numpy.vectorize(_attempt_cast_as_float)
        return to_float(array)


def get_quantize_parent_for_dequantize_node(
    quantized_model: ModelProto, dequantize_node: NodeProto
) -> Union[NodeProto, None]:
    """
    Returns the first quantize node found by traversing the first node input of the
    given de-quantize node's ancestors. All inputs to de-quantize nodes should have
    a quantize node ancestor.

    :param quantized_model: the model the de-quantize node is from
    :param dequantize_node: the node to get an associated quantize node for
    :return: the first quantize node found by traversing the first node input of the
        given de-quantize node's ancestors. If no quantize node is found, returns None
    """
    curr_node = dequantize_node
    while curr_node is not None and curr_node.op_type != "QuantizeLinear":
        input_nodes = get_node_input_nodes(quantized_model, curr_node)
        curr_node = input_nodes[0] if input_nodes else None
    return curr_node


def get_tensor_shape(tensor: onnx.TensorProto) -> List[int]:
    """
    :param tensor: ONNX tensor to get the shape of
    :return: shape of the tensor as a list
    """
    return [dim.dim_value for dim in tensor.type.tensor_type.shape.dim]


def get_tensor_dim_shape(tensor: onnx.TensorProto, dim: int) -> int:
    """
    :param tensor: ONNX tensor to get the shape of a dimension of
    :param dim: dimension index of the tensor to get the shape of
    :return: shape of the tensor at the given dimension
    """
    return tensor.type.tensor_type.shape.dim[dim].dim_value


def set_tensor_dim_shape(tensor: onnx.TensorProto, dim: int, value: int):
    """
    Sets the shape of the tensor at the given dimension to the given value

    :param tensor: ONNX tensor to modify the shape of
    :param dim: dimension index of the tensor to modify the shape of
    :param value: new shape for the given dimension
    """
    tensor.type.tensor_type.shape.dim[dim].dim_value = value
