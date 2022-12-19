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

import numpy
from onnx import ModelProto, helper, numpy_helper

from sparseml.exporters.transforms.onnx_transform import OnnxTransform
from sparseml.exporters.transforms.utils import (
    INITIALIZER_MATCH,
    MatchResult,
    any_of,
    attribute_to_kwarg,
    get_quantization_params,
    get_structural_matches,
    quantize_array,
)
from sparseml.onnx.utils import ONNXGraph, get_init_by_name, get_node_output_nodes


__all__ = ["ConvToQLinearConv"]


class ConvToQLinearConv(OnnxTransform):
    """
    Transforms

    ```
    |     weight (initializer)
    |            |
    | input      Q
    |   |        |
    |   Q/Dq    Dq    bias (optional)
    |       |    |     |
    |           Conv
    |            |
    |           Q/Dq
    ```
    (where `Q` is QuantizeLinear, `Dq` is DequantizeLinear)

    into

    ```
    | input
    |   |
    | QLinearConv
    ```
    """

    def transform(self, model: ModelProto) -> ModelProto:
        graph = ONNXGraph(model)
        matches = get_structural_matches(
            graph,
            op_type="Conv",
            parent_ops=[
                [any_of("QuantizeLinear", "DequantizeLinear")],
                [INITIALIZER_MATCH, "QuantizeLinear", "DequantizeLinear"],
            ],
            children_ops=[[any_of("QuantizeLinear", "DequantizeLinear")]],
        )
        for match in matches:
            self.log_match(match)
            self._transform_match(model, match)
        return model

    def _transform_match(self, model: ModelProto, match: MatchResult):
        conv_node = match.node
        (input_quant,) = match.parents[0]
        _, weight_quant, weight_dequant = match.parents[1]
        (output_quant,) = match.children[0]

        weight_quantize_params = get_quantization_params(
            model, weight_quant, include_target=True
        )
        # sanity check - matching should ensure this
        assert weight_quantize_params.target is not None

        # can fold the input/output quant ops if they are trivial
        fold_input_quant = input_quant.op_type == "DequantizeLinear"
        fold_output_quant = output_quant.op_type == "QuantizeLinear"

        # quantize weight
        quantized_weight = quantize_array(
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
        qconv_input = input_quant.input[0] if fold_input_quant else conv_node.input[0]
        qconv_inputs = [
            qconv_input,  # x
            input_quant.input[1],  # x_scale
            input_quant.input[2],  # x_zero_point
            quantized_weight_name,  # w
            weight_quant.input[1],  # w_scale
            weight_quant.input[2],  # w_zero_point
            output_quant.input[1],  # y_scale
            output_quant.input[2],  # y_zero_point
        ]

        if len(conv_node.input) > 2:
            bias = get_init_by_name(model, conv_node.input[2])
            if bias is not None:
                # quantize bias and add it to the qconv inputs
                bias = numpy_helper.to_array(bias)
                input_quantize_params = get_quantization_params(
                    model, input_quant, include_target=False
                )
                bias_scale = input_quantize_params.scale * weight_quantize_params.scale
                quantized_bias = quantize_array(bias, bias_scale, 0, numpy.int32)
                quantized_bias_name = f"{conv_node.name}.bias_quantized"
                quantized_bias_initializer = numpy_helper.from_array(
                    quantized_bias, name=quantized_bias_name
                )
                model.graph.initializer.append(quantized_bias_initializer)
                qconv_inputs.append(quantized_bias_name)
            else:
                # bias is not initializer, still need to append though
                qconv_inputs.append(conv_node.input[2])

        qconv_kwargs = {}
        for attribute in conv_node.attribute:
            qconv_kwargs.update(attribute_to_kwarg(attribute))

        # create QLinearConv node and add it to graph
        qconv_output = (
            output_quant.output[0] if fold_output_quant else conv_node.output[0]
        )
        self.add_node_deferred(
            helper.make_node(
                "QLinearConv",
                qconv_inputs,
                [qconv_output],
                name=f"{conv_node.name}_quant",
                **qconv_kwargs,
            )
        )

        # Clean up
        self.delete_node_deferred(conv_node)
        self.delete_node_deferred(weight_dequant)
        self.delete_node_deferred(weight_quant)
        if fold_input_quant and len(get_node_output_nodes(model, input_quant)) <= 1:
            self.delete_node_deferred(input_quant)
        if fold_output_quant:
            self.delete_node_deferred(output_quant)
