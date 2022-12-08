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

from typing import NamedTuple, Union

import numpy
from onnx import ModelProto, NodeProto, numpy_helper

from sparseml.onnx.utils import ONNXGraph, remove_node_and_params_from_graph


__all__ = ["delete_quant_node", "get_quantization_params"]

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
