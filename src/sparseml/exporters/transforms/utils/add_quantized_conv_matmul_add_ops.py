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

from typing import Optional, Tuple

import numpy
import onnx
from onnx import ModelProto, NodeProto, TensorProto, numpy_helper

from sparseml.exporters.transforms.utils.helpers import (
    QuantizationParams,
    attribute_to_kwarg,
    quantize_array,
)


__all__ = ["add_quantized_conv_matmul_add_ops"]


def add_quantized_conv_matmul_add_ops(
    model: ModelProto,
    node: NodeProto,
    input_quantize_node: NodeProto,
    weight_quantize_node: NodeProto,
    input_quantize_params: QuantizationParams,
    weight_quantize_params: QuantizationParams,
    bias_initializer: Optional[TensorProto],
    bias_add_name: str,
    target_output: str,
    transpose_weight: bool,
    output_quantize_node: Optional[NodeProto] = None,
    output_dequantize_node: Optional[NodeProto] = None,
) -> ModelProto:
    """
    Helper function for conversion of qat parameterized gemms, matmuls, or convs to
    conv/matmul integer add blocks.

    Adds new quantized ops to graph, does not perform any checks or deletions
    (should be called by the operator main conversion function)
    """
    node_output_orig = node.output[0]
    if not target_output and (
        any(output.name == node_output_orig for output in model.graph.output)
    ):
        # original node output is a graph output, make that the quant block
        # output target id
        target_output = node_output_orig

    # Quantize weights and add to graph
    quantized_weight_initializer = _quantize_weight_initializer(
        node, weight_quantize_params, transpose_weight
    )
    model.graph.initializer.append(quantized_weight_initializer)

    # Create MatMulInteger/ConvInteger node and add it to graph
    integer_op_node = _create_integer_op_node(
        node,
        input_quantize_node,
        weight_quantize_node,
        quantized_weight_initializer.name,
    )
    model.graph.node.append(integer_op_node)

    if bias_initializer is not None:
        # Add bias + zero point correction; quantize bias and add it to graph
        (
            quantized_bias_initializer,
            quantized_bias_scale,
            quantize_bias_zero_point,
        ) = _quantize_bias(
            node,
            bias_initializer,
            input_quantize_params,
            weight_quantize_params,
            bias_add_name,
        )
        model.graph.initializer.append(quantized_bias_initializer)
        model.graph.initializer.append(quantized_bias_scale)
        model.graph.initializer.append(quantize_bias_zero_point)

        # Create Quantized Bias Add node and add it to graph
        qadd_node = _create_qadd_node(
            node,
            integer_op_output="{}_quant".format(node.output[0]),
            quantized_bias_name=quantized_bias_initializer.name,
            output_quantize_node=output_quantize_node,
        )
        model.graph.node.append(qadd_node)
        mul_input_node_name = qadd_node.name

        # bias has same scale as future rescale op, unless doing channel-wise Conv
        if weight_quantize_params.scale.size > 1 and node.op_type == "Conv":
            # channel-wise Conv
            rescale_scale = _create_rescale_init(
                node, input_quantize_params, weight_quantize_params, reshape=True
            )
            model.graph.initializer.append(rescale_scale)
        else:
            rescale_scale = quantized_bias_scale
    else:
        rescale_scale = _create_rescale_init(
            node, input_quantize_params, weight_quantize_params
        )
        model.graph.initializer.append(rescale_scale)
        # cast node should come directly after quantize op output instead of add
        output_quantize_node = output_quantize_node or integer_op_node
        mul_input_node_name = output_quantize_node.name

    # create Cast node and add it to graph
    cast_node = _create_cast_node(
        quant_add_name="{}_bias_add_quant".format(node.name),
        output_quantize_node=output_quantize_node,
    )
    model.graph.node.append(cast_node)

    # create Mul node for rescale
    mul_node = _create_mul_node(
        cast_node_output=cast_node.output[0],
        rescale_scale_name=rescale_scale.name,
        input_node_name=mul_input_node_name,
        target_output=target_output,
        model=model,
        node_output_orig=node_output_orig,
    )
    model.graph.node.append(mul_node)

    return model


def _create_mul_node(
    cast_node_output: str,
    rescale_scale_name: str,
    input_node_name: str,
    target_output: str,
    model: ModelProto,
    node_output_orig: str,
) -> NodeProto:
    mul_node_inputs = [
        cast_node_output,  # a
        rescale_scale_name,  # b -> rescale factor
    ]
    mul_node_name = "{}_rescale_mul".format(input_node_name)
    if target_output is None:
        target_output = mul_node_name
        # since we skip the add conversion,
        # update model to point all outputs of matmul/conv to the rescale mul
        _update_model_input_id(model, node_output_orig, target_output)
    mul_node = onnx.helper.make_node(
        "Mul",
        mul_node_inputs,
        [target_output],
        mul_node_name,
    )
    return mul_node


def _update_model_input_id(model: ModelProto, old_id: str, new_id: str):
    for node in model.graph.node:
        for idx, input_name in enumerate(node.input):
            if input_name == old_id:
                node.input[idx] = new_id


def _create_cast_node(
    quant_add_name: str, output_quantize_node: Optional[NodeProto] = None
) -> NodeProto:
    quant_add_output = (
        output_quantize_node.output[0]
        if output_quantize_node
        else f"{quant_add_name}_output"
    )

    cast_node_name = "{}_cast".format(quant_add_name)
    cast_node_output = "{}_cast".format(quant_add_output)
    cast_node = onnx.helper.make_node(
        "Cast",
        [quant_add_output],
        [cast_node_output],
        cast_node_name,
        to=getattr(onnx.TensorProto, "FLOAT"),  # get Float32 enum id
    )
    return cast_node


def _create_qadd_node(
    node: NodeProto,
    integer_op_output: str,
    quantized_bias_name: str,
    output_quantize_node: Optional[NodeProto] = False,
) -> NodeProto:
    quant_add_inputs = [
        integer_op_output,  # MatMul/Conv integer outputs (INT32)
        quantized_bias_name,  # Quantized bias (INT32)
    ]

    quant_add_name = "{}_bias_add_quant".format(node.name)
    quant_add_output = (
        output_quantize_node.output[0]
        if output_quantize_node
        else f"{quant_add_name}_output"
    )

    # create Add node and add it to graph
    qadd_node = onnx.helper.make_node(
        "Add",
        quant_add_inputs,
        [quant_add_output],
        quant_add_name,
    )
    return qadd_node


def _create_integer_op_node(
    node: NodeProto,
    input_quantize_node: NodeProto,
    weight_quantize_node: NodeProto,
    quantized_weight_name: str,
) -> NodeProto:

    integer_op_inputs = [
        input_quantize_node.input[0],  # input matrix (replaces previous dequant node)
        quantized_weight_name,  # quantized weight
        input_quantize_node.input[2],  # input zero point
        weight_quantize_node.input[2],  # weight zero point
    ]
    integer_op_output = "{}_quant".format(node.output[0])
    integer_op_name = "{}_quant".format(node.name)

    # create MatMulInteger/ConvInteger node and add it to graph
    if node.op_type == "Conv":
        # get conv attributes as kwargs
        conv_kwargs = {}
        for attribute in node.attribute:
            conv_kwargs.update(attribute_to_kwarg(attribute))

        # create ConvInteger node and add it to graph
        integer_op_node = onnx.helper.make_node(
            "ConvInteger",
            integer_op_inputs,
            [integer_op_output],
            integer_op_name,
            **conv_kwargs,
        )
    else:
        integer_op_node = onnx.helper.make_node(
            "MatMulInteger",
            integer_op_inputs,
            [integer_op_output],
            integer_op_name,
        )

    return integer_op_node


def _quantize_bias(
    node, bias_initializer, input_quantize_params, weight_quantize_params, bias_add_name
) -> Tuple[TensorProto, TensorProto, TensorProto]:
    bias_initializer = numpy_helper.to_array(bias_initializer)
    bias_scale = input_quantize_params.scale * weight_quantize_params.scale
    bias_zero_point = numpy.zeros(bias_scale.shape, dtype=numpy.int32)
    quantized_bias = quantize_array(
        bias_initializer, bias_scale, bias_zero_point, dtype=numpy.int32
    )
    if node.op_type == "Conv" and len(quantized_bias.shape) == 1:
        # reshape for bias add broadcasting
        quantized_bias = quantized_bias.reshape(1, quantized_bias.shape[0], 1, 1)

    quantized_bias_name = "{}.bias_quantized".format(bias_add_name)
    quantized_bias_initializer = numpy_helper.from_array(
        quantized_bias, name=quantized_bias_name
    )

    quantized_bias_scale_name = "{}.scale".format(quantized_bias_name)
    quantized_bias_zero_point_name = "{}.zero_point".format(quantized_bias_name)

    return (
        quantized_bias_initializer,
        numpy_helper.from_array(
            numpy.asarray(bias_scale), name=quantized_bias_scale_name
        ),
        numpy_helper.from_array(
            numpy.asarray(bias_zero_point, dtype=numpy.uint8),
            name=quantized_bias_zero_point_name,
        ),
    )


def _create_rescale_init(
    node, input_quantize_params, weight_quantize_params, reshape=False
) -> TensorProto:
    output_scale = input_quantize_params.scale * weight_quantize_params.scale
    if reshape:  # for channel-wise Conv
        output_scale = output_scale.reshape(1, output_scale.shape[0], 1, 1)
    return numpy_helper.from_array(
        numpy.asarray(output_scale), name=f"{node.name}_quant.rescale.scale"
    )


def _quantize_weight_initializer(
    node: NodeProto,
    weight_quantize_params: QuantizationParams,
    transpose_weight: bool = False,
) -> TensorProto:
    quantized_weight = quantize_array(
        weight_quantize_params.target,
        weight_quantize_params.scale,
        weight_quantize_params.zero_point,
        weight_quantize_params.zero_point.dtype,
    )
    if transpose_weight:
        quantized_weight = quantized_weight.transpose()
    quantized_weight_name = "{}.weight_quantized".format(node.name)
    quantized_weight_initializer = numpy_helper.from_array(
        quantized_weight, name=quantized_weight_name
    )
    return quantized_weight_initializer
