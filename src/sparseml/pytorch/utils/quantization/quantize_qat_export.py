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
Helper functions for parsing an exported pytorch model trained with
quantization aware training.
"""


import logging
from collections import defaultdict
from copy import deepcopy
from typing import Any, NamedTuple, Optional, Union

import numpy
import onnx
from onnx import ModelProto, NodeProto, numpy_helper

from sparseml.onnx.utils import (
    ONNXGraph,
    get_batch_norm_params,
    get_init_by_name,
    get_node_attributes,
    get_node_output_nodes,
    get_nodes_by_output_id,
    quantize_resnet_identity_add_inputs,
    quantized_residual_add_optim,
    remove_node_and_params_from_graph,
    swap_node_output,
    update_model_param,
)


__all__ = [
    "get_quantization_params",
    "QuantizationParams",
    "quantize_torch_qat_export",
    "skip_onnx_input_quantize",
]


_LOGGER = logging.getLogger(__name__)


"""
Named tuple object to represent scale/zero point values for quantizing tenors
"""
QuantizationParams = NamedTuple(
    "QuantizationParams",
    [("scale", float), ("zero_point", int), ("target", Union[numpy.ndarray, None])],
)


_QUANTIZE_OP_NAMES = ["QuantizeLinear", "DequantizeLinear"]

_QLINEAR_OP_NAMES = ["QLinearConv", "QLinearMatMul", "QLinearAdd"]


def get_quantization_params(
    model: ModelProto, node: NodeProto, include_target: bool = False
) -> QuantizationParams:
    """
    :param model: ONNX model to read from
    :param node: A QuantizeLinear or DequantizeLinear Node
    :param include_target: Set True include quantization target. If False,
        target value will be returned as None. Default is None
    :return: QuantizationParams object with scale and zero point, will include the
         quantization target if it is an initializer otherwise target will be None
    """
    assert (
        node.op_type in _QUANTIZE_OP_NAMES
    ), "Op Type must be either QuantizeLinear or DequantizeLinear, found {} ".format(
        node.op_type
    )

    scale = get_init_by_name(model, node.input[1])
    if scale is None:
        scale_const = get_nodes_by_output_id(model, node.input[1])
        if scale_const:
            scale = scale_const[0].attribute[0].t
    assert scale, "Quantization scale {} not found".format(node.input[1])

    zero_point = get_init_by_name(model, node.input[2])
    if zero_point is None:
        zero_point_const = get_nodes_by_output_id(model, node.input[2])
        if zero_point_const:
            zero_point = zero_point_const[0].attribute[0].t
    assert zero_point, "Quantization zero point {} not found".format(node.input[2])

    scale = numpy_helper.to_array(scale)
    zero_point = numpy_helper.to_array(zero_point)

    target = None
    if include_target:
        target = get_init_by_name(model, node.input[0])
        if target is not None:
            target = numpy_helper.to_array(target)

    return QuantizationParams(scale=scale, zero_point=zero_point, target=target)


def delete_quant_node(model: ModelProto, node: NodeProto, keep_params: bool = False):
    """
    Deletes a QuantizeLinear or DequantizeLinear and its parameters from the model
    :param model: ONNX model to modify
    :param node: the QuantizeLinear or DequantizeLinear node to delete
    :param keep_params: set true to not delete scale and zero point parameters stored
        in the graph
    """
    assert (
        node.op_type in _QUANTIZE_OP_NAMES
    ), "Op Type must be either QuantizeLinear or DequantizeLinear, found {} ".format(
        node.op_type
    )
    if keep_params:
        del node.input[2]  # delete reference to zero point
        del node.input[1]  # delete reference to scale
    remove_node_and_params_from_graph(model, node)


def _get_single_node_child(
    model: ModelProto, node: NodeProto
) -> Union[NodeProto, None]:
    # return child of input node if it only has one child, otherwise return None
    children = get_node_output_nodes(model, node)
    return children[0] if len(children) == 1 else None


def _get_single_node_parent(
    model: ModelProto, node: NodeProto, input_idx: int
) -> Union[NodeProto, None]:
    # return parent of input node if it only has one parent, otherwise return None
    parent = get_nodes_by_output_id(model, node.input[input_idx])
    return parent[0] if len(parent) == 1 else None


def _fold_conv_bn_bias(model: ModelProto, conv_node: NodeProto, bn_node: NodeProto):
    # fold bias into conv from bn then delete bn node
    bn_params = get_batch_norm_params(model, bn_node)
    variance_term = 1 / numpy.sqrt(bn_params.var + bn_params.epsilon)
    folded_bias = (
        -1.0 * bn_params.mean * variance_term * bn_params.scale + bn_params.bias
    )
    folded_bias = folded_bias.astype(numpy.float32)

    bias_name = conv_node.name + ".bias"
    conv_node.input.append(bias_name)
    update_model_param(model, bias_name, folded_bias)

    # forward conv output to bn children
    swap_node_output(conv_node, bn_node.output[0])
    # remove bn from graph
    remove_node_and_params_from_graph(model, bn_node)


def _fold_qat_conv_bns(model: ModelProto):
    # conv weight should already be folded in quantize linear
    # remove the that div undos the weight folding
    # fold bn into conv bias and remove bn node
    # (Conv -> Div -> BN) -> Conv
    for conv_node in model.graph.node:
        if conv_node.op_type != "Conv" or len(conv_node.input) > 2:
            # not conv node or conv node already has bias
            continue
        div_node = _get_single_node_child(model, conv_node)
        if not div_node or div_node.op_type != "Div":
            continue
        bn_node = _get_single_node_child(model, div_node)
        if not bn_node or bn_node.op_type != "BatchNormalization":
            continue

        # forward conv output to div children
        swap_node_output(conv_node, div_node.output[0])
        # remove div from graph
        remove_node_and_params_from_graph(model, div_node)
        # fold bn into conv bias and remove bn
        _fold_conv_bn_bias(model, conv_node, bn_node)


def _fold_relu_quants(model: ModelProto):
    # delete relu nodes that feed directly into quantize nodes with a zero point of 0
    for relu_node in model.graph.node:
        if relu_node.op_type != "Relu":
            continue
        relu_children = get_node_output_nodes(model, relu_node)
        if not relu_children or any(
            node.op_type != "QuantizeLinear" for node in relu_children
        ):  # skip if any child is not a quantize node
            continue
        quantize_params = [
            get_quantization_params(model, quant_node) for quant_node in relu_children
        ]
        if any(params.zero_point != 0 for params in quantize_params):
            # skip if activation zero point does not match relu threshold of 0
            continue

        # set all child input nodes to the relu node input
        for quant_node in relu_children:
            quant_node.input[0] = relu_node.input[0]
        # delete relu node
        remove_node_and_params_from_graph(model, relu_node)


def _convert_single_constants_to_initializers(model: ModelProto):
    non_single_constant_nodes = []  # list of nodes to keep
    for node in model.graph.node:
        if node.op_type != "Constant" or len(node.attribute) != 1:
            non_single_constant_nodes.append(node)
            continue  # skip non-constants, and constants with multiple tensors

        # create initializer
        const_array = numpy_helper.to_array(node.attribute[0].t)
        # convert int8 -> uint8
        if const_array.dtype == numpy.int8:
            const_array = const_array + 128
            const_array = const_array.astype(numpy.uint8)
        # add named tensor to initializer list
        initializer = numpy_helper.from_array(const_array, name=node.output[0])
        model.graph.initializer.append(initializer)
    # bulk remove all converted constants by overwriting node list
    model.graph.ClearField("node")
    model.graph.node.extend(non_single_constant_nodes)


def _delete_repeated_qat_blocks(model: ModelProto):
    # removes repeated qat quant/dequant blocks with the same parameters
    # (Quant -> Dequant -> Quant -> Dequant) -> (Quant -> Dequant)
    graph = ONNXGraph(model)
    nodes_to_delete = []
    quant_nodes = [n for n in model.graph.node if n.op_type == "QuantizeLinear"]
    for quant_node_1 in quant_nodes:
        dequant_node_1 = graph.get_node_single_child(quant_node_1)
        if not dequant_node_1 or dequant_node_1.op_type != "DequantizeLinear":
            continue
        quant_node_2 = graph.get_node_single_child(dequant_node_1)
        if not quant_node_2 or quant_node_2.op_type != "QuantizeLinear":
            continue
        dequant_node_2 = graph.get_node_single_child(quant_node_2)
        if not dequant_node_2 or dequant_node_2.op_type != "DequantizeLinear":
            continue

        # forward first qat block input to that of the second
        quant_node_2.input[0] = quant_node_1.input[0]

        # remove repeated quant/dequant block
        nodes_to_delete.append(quant_node_1)
        nodes_to_delete.append(dequant_node_1)

    for n in nodes_to_delete:
        delete_quant_node(model, n)


def _attribute_to_kwarg(attribute: onnx.AttributeProto):
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


def _quantize_array(
    array: numpy.ndarray, scale: float, zero_point: int, dtype: Any = numpy.uint8
) -> numpy.ndarray:
    dmin = numpy.iinfo(dtype).min
    dmax = numpy.iinfo(dtype).max
    return ((array / scale).round() + zero_point).clip(dmin, dmax).astype(dtype)


def _convert_quantizable_conv(
    model: ModelProto,
    conv_node: NodeProto,
    input_quantize_node: NodeProto,
    weight_dequantize_node: NodeProto,
    weight_quantize_node: NodeProto,
    output_quantize_node: NodeProto,
):
    weight_quantize_params = get_quantization_params(
        model, weight_quantize_node, include_target=True
    )
    if weight_quantize_params.target is None:
        # weight initializer not included
        return

    # can fold the input/output quant ops if they are trivial
    fold_input_quant = input_quantize_node.op_type == "DequantizeLinear"
    fold_output_quant = output_quantize_node.op_type == "QuantizeLinear"

    # quantize weight
    quantized_weight = _quantize_array(
        weight_quantize_params.target,
        weight_quantize_params.scale,
        weight_quantize_params.zero_point,
        weight_quantize_params.zero_point.dtype,
    )
    quantized_weight_name = "{}.weight_quantized".format(conv_node.name)
    quantized_weight_initializer = numpy_helper.from_array(
        quantized_weight, name=quantized_weight_name
    )
    model.graph.initializer.append(quantized_weight_initializer)

    # get qconv inputs and outputs
    qconv_input = (
        input_quantize_node.input[0] if fold_input_quant else conv_node.input[0]
    )
    qconv_inputs = [
        qconv_input,  # x
        input_quantize_node.input[1],  # x_scale
        input_quantize_node.input[2],  # x_zero_point
        quantized_weight_name,  # w
        weight_quantize_node.input[1],  # w_scale
        weight_quantize_node.input[2],  # w_zero_point
        output_quantize_node.input[1],  # y_scale
        output_quantize_node.input[2],  # y_zero_point
    ]

    if len(conv_node.input) > 2:
        bias = get_init_by_name(model, conv_node.input[2])
        if bias is not None:
            # quantize bias and add it to the qconv inputs
            bias = numpy_helper.to_array(bias)
            input_quantize_params = get_quantization_params(
                model, input_quantize_node, include_target=False
            )
            bias_scale = input_quantize_params.scale * weight_quantize_params.scale
            quantized_bias = _quantize_array(bias, bias_scale, 0, numpy.int32)
            quantized_bias_name = "{}.bias_quantized".format(conv_node.name)
            quantized_bias_initializer = numpy_helper.from_array(
                quantized_bias, name=quantized_bias_name
            )
            model.graph.initializer.append(quantized_bias_initializer)
            qconv_inputs.append(quantized_bias_name)

    qconv_output = (
        output_quantize_node.output[0] if fold_output_quant else conv_node.output[0]
    )
    qconv_name = "{}_quant".format(conv_node.name)
    qconv_kwargs = {}
    for attribute in conv_node.attribute:
        qconv_kwargs.update(_attribute_to_kwarg(attribute))

    # create qconv node and add it to graph
    qconv_node = onnx.helper.make_node(
        "QLinearConv", qconv_inputs, [qconv_output], qconv_name, **qconv_kwargs
    )
    model.graph.node.append(qconv_node)

    # delete original conv and folded quantization ops
    remove_node_and_params_from_graph(model, conv_node)
    delete_quant_node(model, weight_dequantize_node, keep_params=False)
    delete_quant_node(model, weight_quantize_node, keep_params=True)
    if fold_input_quant and len(get_node_output_nodes(model, input_quantize_node)) <= 1:
        # fold if this conv is the only node that reads from this quant op
        delete_quant_node(model, input_quantize_node, keep_params=True)
    if fold_output_quant:
        delete_quant_node(model, output_quantize_node, keep_params=True)


def _convert_quantizable_gemm(
    model: ModelProto,
    gemm_node: NodeProto,
    input_quantize_node: NodeProto,
    weight_dequantize_node: NodeProto,
    weight_quantize_node: NodeProto,
    output_quantize_node: NodeProto,
):
    # Gemm -> (QLinearMatMul -> Add(bias))
    weight_quantize_params = get_quantization_params(
        model, weight_quantize_node, include_target=True
    )
    if weight_quantize_params.target is None:
        # weight initializer not included
        return

    gemm_attributes = get_node_attributes(gemm_node)
    if any(float(attribute) != 1.0 for attribute in gemm_attributes.values()):
        # can only handle Gemm operations without alpha/beta/transB set
        return

    # can fold the input/output quant ops if they are trivial
    fold_input_quant = input_quantize_node.op_type == "DequantizeLinear"
    fold_output_quant = output_quantize_node.op_type == "QuantizeLinear"

    # quantize weight
    quantized_weight = _quantize_array(
        weight_quantize_params.target,
        weight_quantize_params.scale,
        weight_quantize_params.zero_point,
    )
    quantized_weight = quantized_weight.transpose()  # Gemm has implicit transpose
    quantized_weight_name = "{}.weight_quantized".format(gemm_node.name)
    quantized_weight_initializer = numpy_helper.from_array(
        quantized_weight, name=quantized_weight_name
    )
    model.graph.initializer.append(quantized_weight_initializer)

    # get qmatmul inputs and outputs
    qmatmul_input = (
        input_quantize_node.input[0] if fold_input_quant else gemm_node.input[0]
    )
    qmatmul_inputs = [
        qmatmul_input,  # x
        input_quantize_node.input[1],  # x_scale
        input_quantize_node.input[2],  # x_zero_point
        quantized_weight_name,  # w
        weight_quantize_node.input[1],  # w_scale
        weight_quantize_node.input[2],  # w_zero_point
        output_quantize_node.input[1],  # y_scale
        output_quantize_node.input[2],  # y_zero_point
    ]

    qmatmul_output = (
        output_quantize_node.output[0] if fold_output_quant else gemm_node.output[0]
    )
    qmatmul_name = "{}_quant".format(gemm_node.name)

    # create qmatmul node and add it to graph
    qmatmul_node = onnx.helper.make_node(
        "QLinearMatMul",
        qmatmul_inputs,
        [qmatmul_output],
        qmatmul_name,
    )
    model.graph.node.append(qmatmul_node)

    # delete folded quantization ops
    delete_quant_node(model, weight_dequantize_node, keep_params=False)
    delete_quant_node(model, weight_quantize_node, keep_params=True)
    if fold_input_quant and len(get_node_output_nodes(model, input_quantize_node)) <= 1:
        # fold if this gemm is the only node that reads from this quant op
        delete_quant_node(model, input_quantize_node, keep_params=True)
    if fold_output_quant:
        delete_quant_node(model, output_quantize_node, keep_params=True)

    if len(gemm_node.input) > 2:
        # add bias term following FC in the graph
        qmatmul_child_node = get_node_output_nodes(model, qmatmul_node)
        assert qmatmul_child_node, "QLinearMatMul node must have an output in the graph"
        dequant_output_name = "{}_dequantized".format(qmatmul_name)
        if qmatmul_child_node[0].op_type == "DequantizeLinear":
            qmatmul_dequantize_node = qmatmul_child_node[0]
            # create hidden output layer for bias add
            add_output_name = qmatmul_dequantize_node.output[0]
            swap_node_output(qmatmul_dequantize_node, dequant_output_name)
        else:
            # inject dequantize op for matmul
            qmatmul_output_name = "{}_output".format(qmatmul_name)
            swap_node_output(qmatmul_node, qmatmul_output_name)
            qmatmul_dequantize_node = onnx.helper.make_node(
                "DequantizeLinear",
                [
                    qmatmul_output_name,  # input
                    output_quantize_node.input[1],  # scale
                    output_quantize_node.input[2],  # zero point
                ],
                [dequant_output_name],
                "{}_dequantize".format(qmatmul_name),
            )
            model.graph.node.append(qmatmul_dequantize_node)
            add_output_name = qmatmul_output  # original qmatmul output name
        # inject bias op for dequantized matmul output
        qmatmul_bias_add_node = onnx.helper.make_node(
            "Add",
            [
                qmatmul_dequantize_node.output[0],  # add input
                gemm_node.input[2],  # Gemm bias
            ],
            [add_output_name],
            "{}_bias_add".format(gemm_node.name),
        )
        model.graph.node.append(qmatmul_bias_add_node)

        # delete original Gemm node
        params_to_keep = [gemm_node.input[2]] if len(gemm_node.input) > 1 else []
        remove_node_and_params_from_graph(model, gemm_node, keep_params=params_to_keep)


def _convert_quantizable_matmul_and_add(model: ModelProto):
    """
    A pass for converting a MatMul with kernel and bias into a quantized representation

    | Starting with:
    |          INPUT         QuantizeLinear (with constant kernel)
    |            |               |
    |     QuantizeLinear     DequantizeLinear
    |            |               |
    |     DequantizeLinear   Transpose
    |                  |      |
    |                   MatMul
    |                     |
    |                    Add (with constant bias)
    |                     |
    |               QuantizeLinear
    |                     |
    |              DequantizeLinear
    |                     |
    |                  OUTPUT
    | We end up converting to:
    |       INPUT
    |         |
    |     QuantizeLinear
    |         |
    |     QLinearMatMul (with constant kernel)
    |         |
    |     QLinearAdd (with constant bias)
    |         |
    |     DequantizeLinear
    |         |
    |       OUTPUT
    """
    conversion_count = 0
    matmul_nodes = [n for n in model.graph.node if n.op_type in ["MatMul"]]
    for matmul_node in matmul_nodes:
        graph = ONNXGraph(model)
        #############
        # Matching
        #############
        weight_transpose_node = graph.get_node_single_parent(matmul_node, 1)
        if not weight_transpose_node or weight_transpose_node.op_type != "Transpose":
            continue

        weight_dequantize_node = graph.get_node_single_parent(weight_transpose_node, 0)
        if (
            not weight_dequantize_node
            or weight_dequantize_node.op_type != "DequantizeLinear"
        ):
            continue
        weight_quantize_node = graph.get_node_single_parent(weight_dequantize_node, 0)
        if not weight_quantize_node or weight_quantize_node.op_type != "QuantizeLinear":
            continue

        input_quantize_node = graph.get_node_single_parent(matmul_node, 0)
        if (
            not input_quantize_node
            or input_quantize_node.op_type not in _QUANTIZE_OP_NAMES
        ):
            continue

        bias_add_node = graph.get_node_single_child(matmul_node)
        if not bias_add_node or bias_add_node.op_type != "Add":
            continue
        output_quantize_node = graph.get_node_single_child(bias_add_node)
        if (
            not output_quantize_node
            or output_quantize_node.op_type not in _QUANTIZE_OP_NAMES
        ):
            continue

        input_quantize_params = get_quantization_params(
            model, input_quantize_node, include_target=False
        )
        weight_quantize_params = get_quantization_params(
            model, weight_quantize_node, include_target=True
        )
        if weight_quantize_params.target is None:
            # weight initializer not included
            continue
        if input_quantize_node.op_type != "DequantizeLinear":
            continue
        if output_quantize_node.op_type != "QuantizeLinear":
            continue
        bias_initializer = get_init_by_name(model, bias_add_node.input[1])
        if bias_initializer is None:
            continue

        _LOGGER.debug(f"Matched quantizable MatMul weight and bias: {matmul_node.name}")

        #############
        # Conversion
        #############
        # quantize weight
        quantized_weight = _quantize_array(
            weight_quantize_params.target,
            weight_quantize_params.scale,
            weight_quantize_params.zero_point,
        )
        quantized_weight = quantized_weight.transpose()  # Gemm has implicit transpose
        quantized_weight_name = "{}.weight_quantized".format(matmul_node.name)
        quantized_weight_initializer = numpy_helper.from_array(
            quantized_weight, name=quantized_weight_name
        )
        model.graph.initializer.append(quantized_weight_initializer)

        # QLinearMatMul
        # get qmatmul inputs and outputs
        qmatmul_input = input_quantize_node.input[0]
        qmatmul_inputs = [
            qmatmul_input,  # x
            input_quantize_node.input[1],  # x_scale
            input_quantize_node.input[2],  # x_zero_point
            quantized_weight_name,  # w
            weight_quantize_node.input[1],  # w_scale
            weight_quantize_node.input[2],  # w_zero_point
            output_quantize_node.input[1],  # y_scale
            output_quantize_node.input[2],  # y_zero_point
        ]
        qmatmul_output = matmul_node.output[0]
        qmatmul_name = "{}_quant".format(matmul_node.name)

        # create qmatmul node and add it to graph
        qmatmul_node = onnx.helper.make_node(
            "QLinearMatMul",
            qmatmul_inputs,
            [qmatmul_output],
            qmatmul_name,
        )
        model.graph.node.append(qmatmul_node)

        # QLinearAdd
        # quantize bias
        bias_initializer = numpy_helper.to_array(bias_initializer)
        bias_scale = input_quantize_params.scale * weight_quantize_params.scale
        bias_zero_point = 0
        quantized_bias = _quantize_array(bias_initializer, bias_scale, bias_zero_point)
        quantized_bias_name = "{}.bias_quantized".format(bias_add_node.name)
        quantized_bias_initializer = numpy_helper.from_array(
            quantized_bias, name=quantized_bias_name
        )
        model.graph.initializer.append(quantized_bias_initializer)
        quantized_bias_scale_name = "{}.scale".format(quantized_bias_name)
        model.graph.initializer.append(
            numpy_helper.from_array(
                numpy.asarray(bias_scale), name=quantized_bias_scale_name
            )
        )
        quantized_bias_zero_point_name = "{}.zero_point".format(quantized_bias_name)
        model.graph.initializer.append(
            numpy_helper.from_array(
                numpy.asarray(bias_zero_point, dtype=numpy.uint8),
                name=quantized_bias_zero_point_name,
            )
        )

        # get qadd inputs and outputs
        qadd_input = qmatmul_output
        qadd_inputs = [
            qadd_input,  # x
            output_quantize_node.input[1],  # x_scale
            output_quantize_node.input[2],  # x_zero_point
            quantized_bias_name,  # b
            quantized_bias_scale_name,  # b_scale
            quantized_bias_zero_point_name,  # b_zero_point
            output_quantize_node.input[1],  # y_scale
            output_quantize_node.input[2],  # y_zero_point
        ]
        qadd_output = output_quantize_node.output[0]
        qadd_name = "{}_quant".format(bias_add_node.name)
        kwargs = {"domain": "com.microsoft"}
        # create qlinearadd node and add it to graph
        qadd_node = onnx.helper.make_node(
            "QLinearAdd",
            qadd_inputs,
            [qadd_output],
            qadd_name,
            **kwargs,
        )
        model.graph.node.append(qadd_node)

        # Cleanup
        # delete folded quantization ops
        delete_quant_node(model, weight_dequantize_node, keep_params=False)
        delete_quant_node(model, weight_quantize_node, keep_params=True)
        remove_node_and_params_from_graph(model, weight_transpose_node)
        delete_quant_node(model, input_quantize_node, keep_params=True)
        delete_quant_node(model, output_quantize_node, keep_params=True)

        # delete original Gemm node
        remove_node_and_params_from_graph(model, matmul_node, keep_params=None)
        # delete original Add node
        remove_node_and_params_from_graph(model, bias_add_node, keep_params=None)

        conversion_count += 1

    if matmul_nodes:
        _LOGGER.info(
            f"Converted {conversion_count} quantizable MatMul ops with weight and bias "
            "to QLinearMatMul and QLinearAdd"
        )


def _convert_quantizable_ops(model: ModelProto):
    quantizable_nodes = [n for n in model.graph.node if n.op_type in ["Conv", "Gemm"]]
    for quantizable_node in quantizable_nodes:
        graph = ONNXGraph(model)

        weight_dequant = graph.get_node_single_parent(quantizable_node, 1)
        if not weight_dequant or weight_dequant.op_type != "DequantizeLinear":
            continue
        weight_quant = graph.get_node_single_parent(weight_dequant, 0)
        if not weight_quant or weight_quant.op_type != "QuantizeLinear":
            continue

        input_quant = graph.get_node_single_parent(quantizable_node, 0)
        if not input_quant or input_quant.op_type not in _QUANTIZE_OP_NAMES:
            continue

        output_quant = graph.get_node_single_child(quantizable_node)
        if not output_quant or output_quant.op_type not in _QUANTIZE_OP_NAMES:
            continue

        if quantizable_node.op_type == "Conv":
            _convert_quantizable_conv(
                model,
                quantizable_node,
                input_quant,
                weight_dequant,
                weight_quant,
                output_quant,
            )

        if quantizable_node.op_type == "Gemm":
            _convert_quantizable_gemm(
                model,
                quantizable_node,
                input_quant,
                weight_dequant,
                weight_quant,
                output_quant,
            )


def _replace_input_id_model(model: ModelProto, old_id: str, new_id: str):
    for node in model.graph.node:
        for idx, inp in enumerate(node.input):
            if inp == old_id:
                node.input[idx] = new_id


def _remove_duplicate_quantize_ops(model: ModelProto):
    quantize_ops_by_input = defaultdict(list)
    for node in model.graph.node:
        if node.op_type == "QuantizeLinear":
            quantize_ops_by_input[node.input[0]].append(node)

    for quantize_op_group in quantize_ops_by_input.values():
        if len(quantize_op_group) == 1:
            continue
        keep_node = quantize_op_group[0]
        remove_nodes = quantize_op_group[1:]
        for remove_node in remove_nodes:
            _replace_input_id_model(model, remove_node.output[0], keep_node.output[0])
            remove_node_and_params_from_graph(model, remove_node)


def _cleanup_unused_quants(model: ModelProto):
    """
    A pass for removing unused Quantize->Dequantize blocks.
    This should be called at the end of conversion, once all of the conversions
    to quantized operators has been tried.
    Example:
    op -> QuantizeLinear -> DequantizeLinear -> non-quantized op
    => op -> non-quantized operator
    """
    graph = ONNXGraph(model)
    nodes_to_delete = []
    quant_nodes = [n for n in model.graph.node if n.op_type == "QuantizeLinear"]
    for quant_node in quant_nodes:
        dequant_node = graph.get_node_single_child(quant_node)
        if not dequant_node or dequant_node.op_type != "DequantizeLinear":
            continue

        removable = True
        dequant_children = graph.get_node_children(dequant_node)
        for child in dequant_children:
            if isinstance(child, onnx.NodeProto) and child.op_type in _QLINEAR_OP_NAMES:
                removable = False
        if not removable:
            continue

        # Forward QuantizeLinear input to DequantizeLinear output
        for child in dequant_children:
            _replace_input_id_model(model, dequant_node.output[0], quant_node.input[0])

        # Remove QuantizeLinear->DequantizeLinear block
        nodes_to_delete.append(quant_node)
        nodes_to_delete.append(dequant_node)

    for n in nodes_to_delete:
        delete_quant_node(model, n)


def quantize_torch_qat_export(
    model: Union[ModelProto, str],
    output_file_path: Union[str, None] = None,
    inplace: bool = True,
) -> ModelProto:
    """
    :param model: The model to convert, or a file path to it
    :param output_file_path: File path to save the converted model to
    :param inplace: If true, does conversion of model in place. Default is true
    :return: Converts a model exported from a torch QAT session from a QAT graph with
        fake quantize ops surrounding operations to a quantized graph with quantized
        operations. All quantized Convs and FC inputs and outputs be surrounded by
        fake quantize ops
    """
    if isinstance(model, str):
        model = onnx.load(model)

    if not inplace:
        model = deepcopy(model)

    _fold_qat_conv_bns(model)
    _fold_relu_quants(model)
    _convert_single_constants_to_initializers(model)
    _delete_repeated_qat_blocks(model)
    _convert_quantizable_matmul_and_add(model)
    _convert_quantizable_ops(model)
    quantize_resnet_identity_add_inputs(model)
    quantized_residual_add_optim(model)
    _remove_duplicate_quantize_ops(model)
    _cleanup_unused_quants(model)
    ONNXGraph(model).sort_nodes_topologically()

    if output_file_path:
        onnx.save(model, output_file_path)

    return model


def _skip_input_quantize(model: ModelProto) -> Optional[str]:
    if (
        len(model.graph.input) != 1
        or model.graph.input[0].type.tensor_type.elem_type != 1
    ):
        # more than 1 input or input is not FP32
        return (
            "Not modifying ONNX graph inputs - either graph has more than one "
            "input or input type is not FP32"
        )

    input_node = model.graph.input[0]
    input_children = [
        node for node in model.graph.node if input_node.name in node.input
    ]
    if not all(node.op_type == "QuantizeLinear" for node in input_children):
        return (
            "Not modifying ONNX graph inputs - only QuantizeLinear nodes may follow"
            "the FP32 input tensor in original graph, prior to converting to uint8"
        )

    graph = ONNXGraph(model)
    for quantize_node in input_children:
        quantize_children = graph.get_node_children(quantize_node)
        quantize_node_id = quantize_node.output[0]
        for child_node in quantize_children:
            input_idx = [
                idx
                for idx, inp in enumerate(child_node.input)
                if inp == quantize_node_id
            ]
            if not input_idx:
                continue
            input_idx = input_idx[0]
            graph.update_node_input(child_node, input_node.name, input_idx)
            _LOGGER.debug(
                f"set node with output id {child_node.output[0]} as initial node in "
                "graph"
            )

    _LOGGER.debug(
        f"deleting QuantizeLinear node(s) with output id(s): "
        f"{[n.output for n in input_children]}"
    )
    graph.delete_nodes(input_children)  # only contains references to the Quantize nodes
    graph.delete_unused_initializers()  # cleanup
    input_node.type.tensor_type.elem_type = 2  # fp32 -> uint8
    _LOGGER.info("Model initial QuantizeLinear node(s) deleted and inputs set to uint8")

    return None


def skip_onnx_input_quantize(
    model: Union[ModelProto, str],
    output_file_path: Union[str, None] = None,
):
    """
    If the given model has a single FP32 input that feeds into a QuantizeLinear
    node, then the input will be changed to uint8 and the QuantizeLinear node will be
    deleted. This enables quantize graphs to take quantized inputs instead of floats.

    If no optimization is made, a RuntimeError will be raised.

    :param model: The model to convert, or a file path to it
    :param output_file_path: File path to save the converted model to
    """
    if isinstance(model, str):
        model = onnx.load(model)

    optim_error_message = _skip_input_quantize(model)

    if optim_error_message:
        raise RuntimeError(optim_error_message)

    if output_file_path:
        onnx.save(model, output_file_path)
