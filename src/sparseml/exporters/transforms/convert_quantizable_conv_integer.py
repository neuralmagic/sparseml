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
from typing import Union

import onnx
from onnx import ModelProto

from sparseml.exporters.transforms.base_transform import BaseTransform
from sparseml.exporters.transforms.utils import (
    INITIALIZER_MATCH,
    MatchResult,
    add_quantized_conv_matmul_add_ops,
    delete_quant_node,
    get_quantization_params,
    get_structural_matches,
)
from sparseml.onnx.utils import (
    ONNXGraph,
    check_load_model,
    remove_node_and_params_from_graph,
    validate_onnx_file,
)


_LOGGER = logging.getLogger(__name__)


def convert_conv_to_quantized(
    model: ModelProto,
    match: MatchResult,
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
    input_quantize_node, input_dequantize_node = match.parents[0]
    weight_init, weight_quantize_node, weight_dequantize_node = match.parents[1]
    (bias_init,) = match.parents[2]

    model = add_quantized_conv_matmul_add_ops(
        model=model,
        node=match.node,
        input_quantize_node=input_quantize_node,
        weight_quantize_node=weight_quantize_node,
        input_quantize_params=get_quantization_params(
            model, input_quantize_node, include_target=True
        ),
        weight_quantize_params=get_quantization_params(
            model, weight_quantize_node, include_target=True
        ),
        bias_initializer=bias_init,
        bias_add_name="{}_bias_add".format(match.node.name),
        target_output=match.node.output[0],
        transpose_weight=False,
    )

    delete_quant_node(model, weight_dequantize_node)
    delete_quant_node(model, weight_quantize_node)
    delete_quant_node(model, input_dequantize_node)

    # only delete input node if the matmul is the only child
    current_graph = ONNXGraph(model)
    if len(current_graph.get_node_children(input_quantize_node)) == 1:
        delete_quant_node(model, input_quantize_node)

    # delete original Conv node
    remove_node_and_params_from_graph(model, match.node)

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
        graph = ONNXGraph(model)
        for match in get_structural_matches(
            graph,
            parent_ops=[
                ["QuantizeLinear", "DequantizeLinear"],
                [
                    # weight should be initializer
                    INITIALIZER_MATCH,
                    "QuantizeLinear",
                    "DequantizeLinear",
                ],
                [
                    # bias should be initializer
                    INITIALIZER_MATCH
                ],
            ],
            op_type="Conv",
        ):
            _LOGGER.debug(
                f"Matched quantizable Conv weight and bias: {match.node.name}"
            )
            model = convert_conv_to_quantized(model, match)

            count_converted_nodes += 1

        if count_converted_nodes > 0:
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
