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
from sparseml.exporters.transforms.utils.helpers import (
    attribute_to_kwarg,
    delete_quant_node,
    get_quantization_params,
    quantize_array,
)
from sparseml.exporters.transforms.utils.matching import (
    INITIALIZER_MATCH,
    MatchResult,
    get_structural_matches,
)
from sparseml.onnx.utils.graph_editor import (
    ONNXGraph,
    remove_node_and_params_from_graph,
)
from sparseml.onnx.utils.helpers import get_init_by_name, get_node_output_nodes


class ConvToQLinearConv(OnnxTransform):
    """
    Transforms

    ```
                        weight (initializer)
                        |
    input               QuantizeLinear
    |                   |
    DequantizeLinear    DequantizeLinear
    |                   |
            Conv
            |
            QuantizeLinear
    ```

    into

    `input -> QLinearConv`
    """

    def transform(self, model: ModelProto) -> ModelProto:
        graph = ONNXGraph(model)
        matches = get_structural_matches(
            graph,
            op_type="Conv",
            parent_ops=[
                ["DequantizeLinear"],
                [INITIALIZER_MATCH, "QuantizeLinear", "DequantizeLinear"],
            ],
            children_ops=[["QuantizeLinear"]],
        )
        for match in matches:
            self._do_transform(model, match)
        ONNXGraph(model).sort_nodes_topologically()
        return model

    def _do_transform(self, model: ModelProto, match: MatchResult):
        conv_node = match.node
        (input_dequant,) = match.parents[0]
        weight_init, weight_quant, weight_dequant = match.parents[1]
        (output_quant,) = match.children[0]

        weight_quantize_params = get_quantization_params(
            model, weight_quant, include_target=True
        )
        # sanity check - matching should ensure this
        assert weight_quantize_params.target is not None

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
        qconv_input = input_dequant.input[0]
        qconv_inputs = [
            qconv_input,  # x
            input_dequant.input[1],  # x_scale
            input_dequant.input[2],  # x_zero_point
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
                    model, input_dequant, include_target=False
                )
                bias_scale = input_quantize_params.scale * weight_quantize_params.scale
                quantized_bias = quantize_array(bias, bias_scale, 0, numpy.int32)
                quantized_bias_name = "{}.bias_quantized".format(conv_node.name)
                quantized_bias_initializer = numpy_helper.from_array(
                    quantized_bias, name=quantized_bias_name
                )
                model.graph.initializer.append(quantized_bias_initializer)
                qconv_inputs.append(quantized_bias_name)

        qconv_output = output_quant.output[0]
        qconv_name = "{}_quant".format(conv_node.name)
        qconv_kwargs = {}
        for attribute in conv_node.attribute:
            qconv_kwargs.update(attribute_to_kwarg(attribute))

        # create qconv node and add it to graph
        qconv_node = helper.make_node(
            "QLinearConv", qconv_inputs, [qconv_output], qconv_name, **qconv_kwargs
        )
        model.graph.node.append(qconv_node)

        # delete original conv and folded quantization ops
        remove_node_and_params_from_graph(model, conv_node)
        delete_quant_node(model, weight_dequant)
        delete_quant_node(model, weight_quant, keep_weight=True)
        if len(get_node_output_nodes(model, input_dequant)) <= 1:
            # fold if this conv is the only node that reads from this quant op
            delete_quant_node(model, input_dequant)
        delete_quant_node(model, output_quant)
        return qconv_node
