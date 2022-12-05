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
from sparseml.exporters.transforms.helpers import check_for_sequence_of_parent_nodes, assert_node_type, get_quantization_params
from sparseml.onnx.utils import (
    ONNXGraph,
    check_load_model,
    remove_node_and_params_from_graph,
    validate_onnx_file,
)


_LOGGER = logging.getLogger(__name__)

OPTIONAL_NODES_NAMES = {"Transpose", "Reshape"}

def is_quantizable_conv(conv_node: NodeProto, model: ModelProto) -> bool:
    """
    :param conv_node: A conv_node (i.e. node with op_type == "Conv")
    :return:
    """
    graph = ONNXGraph(model)

    if len(conv_node.input) != 3:
        # this function currently only converts Conv nodes with bias param
        # (i.e. from folded batch norm value)
        return False

    # check the weight nodes
    weight_dequantize_node = graph.get_node_single_parent(conv_node, 1)
    if not assert_node_type(weight_dequantize_node, "DequantizeLinear") or not check_for_sequence_of_parent_nodes(weight_dequantize_node, graph, ["QuantizeLinear"]):
        return False

    # check the input nodes
    input_quantize_node = graph.get_node_single_parent(conv_node, 0)
    if not assert_node_type(input_quantize_node, "DequantizeLinear"):
        return False

    weight_quantize_node = graph.get_node_single_parent(weight_dequantize_node, 0)
    weight_quantize_params = get_quantization_params(model,weight_quantize_node, include_target=True)
    if weight_quantize_params.target is None:
        # weight initializer not included
        return False

    # check the bias
    bias_initializer = graph.get_init_by_name(conv_node.input[2])
    if bias_initializer is None:
        _LOGGER.debug(f"Unable to find bias initializer: {conv_node.input[2]}")
        return False

    return True


class ConvertQuantizableConvInteger(BaseTransform):
    """
    A transform that attempts, if possible, to convert Convolution Op with kernel whose activations
    are not necessarily quantized into a ConvInteger followed by a bias add and cast to FP32
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
            if not is_quantizable_conv(conv_node, model):
                continue
            _LOGGER.debug(f"Matched quantizable Conv weight and bias: {conv_node.name}")

            # Convert
            count_converted_nodes += 1

        if conv_nodes:
            _LOGGER.info(
                f"Converted {count_converted_nodes} quantizable Conv ops with weight and bias "
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
