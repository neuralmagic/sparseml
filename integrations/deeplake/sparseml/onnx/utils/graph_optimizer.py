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
Helper functions to optimize ONNX Graphs.
"""


from typing import Tuple, Union

import numpy as np
import onnx

from sparseml.onnx.utils.graph_editor import (
    ONNXGraph,
    remove_node_and_params_from_graph,
    swap_node_output,
    update_model_param,
)
from sparseml.onnx.utils.helpers import (
    BatchNormParams,
    NodeParam,
    conv_node_params,
    get_batch_norm_params,
    get_quantize_parent_for_dequantize_node,
)


__all__ = [
    "fold_conv_bns",
    "quantize_resnet_identity_add_inputs",
]


def _get_folded_conv_params(
    conv_params: Tuple[NodeParam, Union[NodeParam, None]],
    bn_params: BatchNormParams,
) -> Tuple[NodeParam, Union[NodeParam, None]]:
    conv_weight, conv_bias = conv_params
    weight_new, bias_new = None, None
    if bn_params.var is not None and bn_params.epsilon is not None:
        variance_term = 1 / np.sqrt(bn_params.var + bn_params.epsilon)
        # Compute new weight if possible
        if bn_params.scale is not None:
            bn_term = (bn_params.scale * variance_term).reshape(-1, 1, 1, 1)
            weight_new = NodeParam(conv_weight.name, conv_weight.val * bn_term)

        # Compute new bias if possible
        if (
            bn_params.scale is not None
            and bn_params.bias is not None
            and bn_params.mean is not None
        ):
            # if conv bias is not present, default to zeros
            conv_bias = conv_bias or NodeParam(None, np.zeros(bn_params.mean.shape))
            normalized_bias = (conv_bias.val - bn_params.mean) * variance_term
            bias_new = normalized_bias * bn_params.scale + bn_params.bias
        # type check
        if bias_new is not None and weight_new is not None:
            bias_new = NodeParam(conv_bias.name, bias_new.astype(weight_new.val.dtype))

    return weight_new, bias_new


def _fold_conv_bn(
    model: onnx.ModelProto,
    conv_node: onnx.NodeProto,
    bn_node: onnx.NodeProto,
) -> bool:
    """
    Folds the linear operations in bn_node into conv_node
    :param model: The model to fold the node in
    :param conv_node: The conv node to fold in to
    :param bn_node: The batch norm node to fold
    :return: True if the fold succeeded
    """
    conv_params = conv_node_params(model, conv_node)
    bn_params = get_batch_norm_params(model, bn_node)

    folded_weight, folded_bias = _get_folded_conv_params(conv_params, bn_params)

    if folded_weight is not None:
        # Update conv weight to folded and bias if possible
        update_model_param(model, folded_weight.name, folded_weight.val)
        if folded_bias is not None:
            bias_name = folded_bias.name
            if bias_name is None:
                bias_name = folded_weight.name.split("weight")[0] + "bias"
                conv_node.input.append(bias_name)
            update_model_param(model, bias_name, folded_bias.val)
        swap_node_output(
            conv_node, bn_node.output[0]
        )  # forward the folded conv outputs
        remove_node_and_params_from_graph(model, bn_node)  # remove the bn op
        return True
    return False


def fold_conv_bns(onnx_file: Union[str, onnx.ModelProto]) -> onnx.ModelProto:
    """
    When a batch norm op is the only child operator of a conv op, this function
    will fold the batch norm into the conv and return the processed graph

    :param onnx_file: file path to ONNX model to process or in-memory ModelProto
        to be modified in-place
    :return: A loaded ONNX model with BatchNormalization ops folded into Conv ops
        where possible
    """
    model = onnx.load(onnx_file) if isinstance(onnx_file, str) else onnx_file
    conv_nodes = [n for n in model.graph.node if n.op_type == "Conv"]
    graph_modified = False
    for conv_node in conv_nodes:
        conv_output = conv_node.output[0]
        child_nodes = [n for n in model.graph.node if conv_output in n.input]
        # Check if the only child of the conv output is a batch norm op
        if len(child_nodes) == 1 and child_nodes[0].op_type == "BatchNormalization":
            bn_node = child_nodes[0]
            fold_performed = _fold_conv_bn(model, conv_node, bn_node)
            graph_modified = fold_performed or graph_modified
    return model if graph_modified else None


def quantize_resnet_identity_add_inputs(quantized_model: onnx.ModelProto) -> bool:
    """
    To avoid storing the identity value of a ResNet block in fp32, this optimization
    will pass the identity value through the same quantize operation as the ResNet
    block and add a de-quantize operation for the identity before the add.

    Function will match to any add operation whose inputs are the output of a relu
    or add op and a quantize -> de-quantize block that takes the same relu as input.
    Performs this optimization in place.

    :param quantized_model: A loaded quantized model to perform this optimization on
    :return: True if an in-place optimization was made
    """

    add_nodes = [node for node in quantized_model.graph.node if node.op_type == "Add"]
    optimization_made = False
    for add_node in add_nodes:
        graph = ONNXGraph(quantized_model)
        add_inputs = [
            i for i in graph.get_node_parents(add_node) if isinstance(i, onnx.NodeProto)
        ]
        if len(add_inputs) != 2:
            continue
        # extract dequantize input and relu/add input
        dequantize_node = [i for i in add_inputs if i.op_type == "DequantizeLinear"]
        other_input_node = [i for i in add_inputs if i.op_type in ["Add", "Relu"]]
        if not dequantize_node or not other_input_node:  # pattern not matched
            continue
        dequantize_node = dequantize_node[0]  # unwrap
        other_input_node = other_input_node[0]  # unwrap

        quantize_node = get_quantize_parent_for_dequantize_node(
            quantized_model, dequantize_node
        )

        # check that the quantize block takes input from the same relu
        if (
            quantize_node is None
            or quantize_node.input[0] != other_input_node.output[0]
        ):
            continue

        # create de-quantize node for identity
        identity_dequantize_inputs = [quantize_node.output[0]] + quantize_node.input[1:]
        dequantize_identity_output_name = "{}_identity_dequantized".format(
            other_input_node.output[0]
        )
        dequantize_identity_node_name = "{}_identity_dequantized".format(
            other_input_node.output[0]
        )
        identity_dequantize_node = onnx.helper.make_node(
            "DequantizeLinear",
            identity_dequantize_inputs,
            [dequantize_identity_output_name],
            dequantize_identity_node_name,
        )
        quantized_model.graph.node.append(identity_dequantize_node)

        # swap the relu input for the de-quantized identity in the add
        relu_input_idx = [
            i
            for i, inp in enumerate(add_node.input)
            if inp == other_input_node.output[0]
        ][0]
        add_node.input[relu_input_idx] = dequantize_identity_output_name

        optimization_made = True

    return optimization_made


def _make_dequant_node_for_quant(quant_node: onnx.NodeProto) -> onnx.NodeProto:
    return onnx.helper.make_node(
        "DequantizeLinear",
        [quant_node.output[0]] + quant_node.input[1:],  # new inputs
        [f"{quant_node.output[0]}_dequantized"],  # output name
        f"{quant_node.name or quant_node.output[0]}_dequantized",  # node name
    )
