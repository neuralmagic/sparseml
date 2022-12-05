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
from typing import Optional, Tuple, Union

import onnx
from onnx import ModelProto, NodeProto

from sparseml.exporters.transforms import BaseTransform
from sparseml.exporters.transforms.helpers import assert_node_type, delete_quant_node
from sparseml.exporters.transforms.utils import (
    add_quantized_conv_matmul_add_ops,
    get_quantization_params,
)
from sparseml.onnx.utils import (
    ONNXGraph,
    check_load_model,
    remove_node_and_params_from_graph,
    validate_onnx_file,
)


_LOGGER = logging.getLogger(__name__)


def is_quantizable_conv_int(
    conv_node: NodeProto, model: ModelProto
) -> Optional[Tuple[NodeProto, NodeProto, NodeProto]]:
    """
    Checks if a convolution node is quantizable

    :param conv_node: A convolution node (i.e a node with op_type == "Conv")
    :param graph: An ONNX graph that the node belongs to
    :return: One of the following:
        - None if the convolution node is not quantizable
        - otherwise, a tuple of:
            - the input quantize node
            - the quantized weights node
            - the dequantized weights node
    """
    graph = ONNXGraph(model)

    if len(conv_node.input) != 3:
        # this function currently only converts Conv nodes with bias param
        # (i.e. from folded batch norm value)
        return None

    # verify that between the input and the conv node
    # there is a quantize and dequantize node
    weight_dequantize_node = graph.get_node_single_parent(conv_node, 1)
    weight_quantize_node = graph.get_node_single_parent(weight_dequantize_node, 0)
    if not assert_node_type(
        weight_dequantize_node, "DequantizeLinear"
    ) or not assert_node_type(weight_quantize_node, "QuantizeLinear"):
        return None

    # verify that between the input and the conv node
    # there is a quantize and dequantize node
    input_quantize_node = graph.get_node_single_parent(conv_node, 0)
    input_dequantize_node = graph.get_node_single_parent(input_quantize_node, 0)
    if not assert_node_type(
        input_quantize_node, "DequantizeLinear"
    ) or not assert_node_type(input_dequantize_node, "QuantizeLinear"):
        return None

    # verify that the weights and bias have a valid initializer
    weight_quantize_params = get_quantization_params(
        model, weight_quantize_node, include_target=True
    )
    bias_initializer = graph.get_init_by_name(conv_node.input[2])
    if weight_quantize_params.target is None:
        # weight initializer not included
        return None
    if bias_initializer is None:
        # bias initializer not included
        _LOGGER.debug(f"Unable to find bias initializer: {conv_node.input[2]}")
        return None

    return (
        input_quantize_node,
        weight_quantize_node,
        weight_dequantize_node,
    )


def convert_conv_to_quantized(
    model: ModelProto,
    conv_node: NodeProto,
    weight_quantize_node: NodeProto,
    weight_dequantize_node: NodeProto,
    input_quantize_node: NodeProto,
) -> onnx.ModelProto:
    """
    Converts a conv node to a quantized conv node
    and updates the onnx model accordingly

    :param model: ONNX model to convert
    :param conv_node: The convolution node to convert
    :param weight_quantize_node: The quantize node for the weights of the conv node
    :param weight_dequantize_node: The dequantize node for the weights of the conv node
    :param input_quantize_node: The quantize node for the input of the conv node
    :return: The converted ONNX model
    """
    graph = ONNXGraph(model)
    model = add_quantized_conv_matmul_add_ops(
        model=model,
        node=conv_node,
        input_quantize_node=input_quantize_node,
        weight_quantize_node=weight_quantize_node,
        input_quantize_params=get_quantization_params(
            model, input_quantize_node, include_target=True
        ),
        weight_quantize_params=get_quantization_params(
            model, weight_quantize_node, include_target=True
        ),
        bias_initializer=graph.get_init_by_name(conv_node.input[2]),
        bias_add_name="{}_bias_add".format(conv_node.name),
        target_output=conv_node.output[0],
        transpose_weight=False,
    )

    delete_quant_node(model, weight_dequantize_node)
    delete_quant_node(model, weight_quantize_node)

    # only delete input node if the matmul is the only child
    current_graph = ONNXGraph(model)
    if len(current_graph.get_node_children(input_quantize_node)) == 1:
        delete_quant_node(model, input_quantize_node)

    # delete original Conv node
    remove_node_and_params_from_graph(model, conv_node)

    return model


class ConvertQuantizableConvInteger(BaseTransform):
    """
    A transform that attempts, if possible, to convert Convolution Op
    with kernel whose activations are not necessarily quantized into a
    ConvInteger followed by a bias add and cast to FP32.
    This MatMul is the result of quantizing native torch.matmul using QATMatMul

    | Starting with:
    |          INPUT         QuantizeLinear (with constant kernel)
    |            |               |
    |     QuantizeLinear     DequantizeLinear
    |            |               |
    |     DequantizeLinear      |
    |                  |      |
    |                   Conv (with bias)
    |                     |
    |                  OUTPUT
    | We end up converting to:
    |       INPUT
    |         |
    |     QuantizeLinear
    |         |
    |     ConvInteger (with constant uint8 kernel)
    |         |
    |     Add (constant bias + zero point correction)
    |         |
    |     Cast (INT32 -> FP32)
    |         |
    |     Mul (Rescale from bias scale)
    |         |
    |       OUTPUT
    """

    def _transform(self, model: ModelProto) -> ModelProto:
        count_converted_nodes = 0
        conv_nodes_indices, conv_nodes = zip(
            *[
                (idx, node)
                for idx, node in enumerate(model.graph.node)
                if node.op_type == "Conv"
            ]
        )
        for idx, conv_node in zip(conv_nodes_indices, conv_nodes):
            # Check whether there is a match
            result = is_quantizable_conv_int(conv_node, model)
            if result is None:
                continue
            input_quantize_node, weight_quantize_node, weight_dequantize_node = result
            _LOGGER.debug(f"Matched quantizable Conv weight and bias: {conv_node.name}")
            model = convert_conv_to_quantized(
                model,
                conv_node,
                weight_quantize_node,
                weight_dequantize_node,
                input_quantize_node,
            )

            count_converted_nodes += 1

        if conv_nodes:
            _LOGGER.info(
                f"Converted {count_converted_nodes} quantizable "
                f"Conv ops with weight and bias "
                "to ConvInteger and Add"
            )
        return model

    def _validate_input(self, model: ModelProto):
        validate_onnx_file(model)

    def _validate_output(self, model: ModelProto):
        validate_onnx_file(model)

    def apply(self, model: Union[ModelProto, str]) -> ModelProto:
        onnx_model = check_load_model(model)
        self._validate_input(onnx_model)
        onnx_model = self._transform(onnx_model)
        self._validate_output(onnx_model)
        return onnx_model
