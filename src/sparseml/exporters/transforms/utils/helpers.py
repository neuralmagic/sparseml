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


import logging
from typing import Any, List, NamedTuple, Set, Union

import numpy
from onnx import AttributeProto, ModelProto, NodeProto, numpy_helper

from sparseml.onnx.utils import ONNXGraph, remove_node_and_params_from_graph


_LOGGER = logging.getLogger(__name__)

__all__ = [
    "delete_quant_node",
    "get_quantization_params",
    "quantize_array",
    "assert_node_type",
    "QUANTIZE_OP_NAMES",
    "attribute_to_kwarg",
]

QUANTIZE_OP_NAMES = ["QuantizeLinear", "DequantizeLinear"]

QuantizationParams = NamedTuple(
    "QuantizationParams",
    [("scale", float), ("zero_point", int), ("target", Union[numpy.ndarray, None])],
)


def get_quantization_params(
    model: Union[ModelProto, ONNXGraph],
    node: NodeProto,
    include_target: bool = False,
) -> QuantizationParams:
    """
    :param model: ONNX model to read from or ONNXGraph object
    :param node: A QuantizeLinear or DequantizeLinear Node
    :param include_target: Set True include quantization target. If False,
        target value will be returned as None. Default is None
    :return: QuantizationParams object with scale and zero point, will include the
         quantization target if it is an initializer otherwise target will be None
    """
    assert (
        node.op_type in QUANTIZE_OP_NAMES
    ), "Op Type must be either QuantizeLinear or DequantizeLinear, found {} ".format(
        node.op_type
    )

    graph = model if isinstance(model, ONNXGraph) else ONNXGraph(model)

    scale = graph.get_init_by_name(node.input[1])
    if scale is None:
        scale_const = graph.get_node_by_output_id(node.input[1])
        if scale_const:
            scale = scale_const.attribute[0].t
    assert scale, "Quantization scale {} not found".format(node.input[1])

    zero_point = graph.get_init_by_name(node.input[2])
    if zero_point is None:
        zero_point_const = graph.get_node_by_output_id(node.input[2])
        if zero_point_const:
            zero_point = zero_point_const.attribute[0].t
    assert zero_point, "Quantization zero point {} not found".format(node.input[2])

    scale = numpy_helper.to_array(scale)
    zero_point = numpy_helper.to_array(zero_point)

    target = None
    if include_target:
        target = graph.get_init_by_name(node.input[0])
        if target is not None:
            target = numpy_helper.to_array(target)

    return QuantizationParams(scale=scale, zero_point=zero_point, target=target)


def delete_quant_node(
    model: ModelProto,
    node: NodeProto,
    keep_weight: bool = False,
):
    """
    Deletes a QuantizeLinear or DequantizeLinear and its parameters from the model
    :param model: ONNX model to modify
    :param node: the QuantizeLinear or DequantizeLinear node to delete
    :param keep_weight: set true to not delete the weight param possibly stored as an
        initializer to the first input of this node
    """
    assert (
        node.op_type in QUANTIZE_OP_NAMES
    ), "Op Type must be either QuantizeLinear or DequantizeLinear, found {} ".format(
        node.op_type
    )
    if keep_weight:
        del node.input[0]
    remove_node_and_params_from_graph(model, node)


def assert_node_type(node: NodeProto, op: Union[List[str], Set[str], str]) -> bool:
    """
    Checks if a node is of the given op type
    :param node: the node to check
    :param op: the operation type to check for
    :return: True if the node has the given op type, False otherwise
    """
    if node is None:
        return False
    if isinstance(op, str):
        return node.op_type == op
    else:
        return node.op_type in op


def quantize_array(
    array: numpy.ndarray, scale: float, zero_point: int, dtype: Any = numpy.uint8
) -> numpy.ndarray:
    try:
        import torch  # noqa: F401

        if dtype == numpy.uint8:
            tensor_dtype = torch.quint8
        elif dtype == numpy.int8:
            tensor_dtype = torch.qint8
        elif dtype == numpy.int32:
            tensor_dtype = torch.qint32

        tensor = torch.Tensor(array.copy()).to(torch.float32)
        if isinstance(scale, numpy.ndarray):
            scale = scale.item()
        if isinstance(zero_point, numpy.ndarray):
            zero_point = zero_point.item()

        quant_tensor = torch.quantize_per_tensor(
            tensor, scale, zero_point, tensor_dtype
        )
        return quant_tensor.int_repr().numpy()
    except ModuleNotFoundError as err:
        _LOGGER.debug(f"Error: {err}. Defaulting to numpy implementation.")
        dmin = numpy.iinfo(dtype).min
        dmax = numpy.iinfo(dtype).max
        return ((array / scale).round() + zero_point).clip(dmin, dmax).astype(dtype)


def attribute_to_kwarg(attribute: AttributeProto):
    # Adapted from ORT quantize.py
    if attribute.type == 0:
        raise ValueError(
            "attribute {} does not have type specified.".format(attribute.name)
        )

    # Based on attribute type definitions from AttributeProto
    # definition in https://github.com/onnx/onnx/blob/master/onnx/onnx.proto
    if attribute.type == 1:
        value = attribute.f
    elif attribute.type == 2:
        value = attribute.i
    elif attribute.type == 3:
        value = attribute.s
    elif attribute.type == 4:
        value = attribute.t
    elif attribute.type == 5:
        value = attribute.g
    elif attribute.type == 6:
        value = attribute.floats
    elif attribute.type == 7:
        value = attribute.ints
    elif attribute.type == 8:
        value = attribute.strings
    elif attribute.type == 9:
        value = attribute.tensors
    elif attribute.type == 10:
        value = attribute.graphs
    else:
        raise ValueError(
            "attribute {} has unsupported type {}.".format(
                attribute.name, attribute.type
            )
        )

    return {attribute.name: value}
