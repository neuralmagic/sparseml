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
from onnx import ModelProto, numpy_helper

from sparseml.exporters.transforms.onnx_transform import OnnxTransform
from sparseml.exporters.transforms.utils.matching import (
    MatchResult,
    get_structural_matches,
)
from sparseml.onnx.utils import ONNXGraph, get_batch_norm_params, get_init_by_name


__all__ = ["FoldConvDivBn"]


class FoldConvDivBn(OnnxTransform):
    """
    The purpose of this pass is to handle the fake batch norm folding
    that pytorch does. Notably the div node is already folded into
    the batch norm node by pytorch, and the variance from the batch norm
    is also already folded into the conv node. Therefore, the only
    thing remaining is to fold the bias from the batch norm node into
    the Conv node, which this node does.

    Specifically, this transforms

    ```
    | input   weight  bias (optional)
    |      |    |    |
    |         Conv
    |           |
    |         Div
    |           |
    |     BatchNormalization
    ```

    into

    ```
    | input  weight  bias
    |      |   |    |
    |        Conv
    ```
    """

    def transform(self, model: ModelProto) -> ModelProto:
        matches = get_structural_matches(
            ONNXGraph(model),
            op_type="Conv",
            children_ops=[["Div", "BatchNormalization"]],
        )
        for match in matches:
            self.log_match(match)
            self._transform_match(model, match)
        return model

    def _transform_match(self, model: ModelProto, match: MatchResult):
        conv_node = match.node
        div_node, bn_node = match.children[0]

        # get bn params
        bn_params = get_batch_norm_params(model, bn_node)

        # get conv bias or initialize to zeros
        conv_bias = numpy.zeros(bn_params.mean.shape)
        if len(conv_node.input) > 2:
            conv_bias_init = get_init_by_name(model, conv_node.input[2])
            if conv_bias_init is not None:
                conv_bias = numpy_helper.to_array(conv_bias_init)
            else:
                raise ValueError(
                    "Bias was not an initializer, transform executed in wrong order"
                )

        # fold bias into conv from bn then delete bn node
        variance_term = 1 / numpy.sqrt(bn_params.var + bn_params.epsilon)
        normalized_bias = (conv_bias - bn_params.mean) * variance_term
        folded_bias = normalized_bias * bn_params.scale + bn_params.bias
        folded_bias = folded_bias.astype(numpy.float32)

        bias_name = conv_node.name + ".bias"
        if len(conv_node.input) > 2:
            conv_node.input[2] = bias_name
        else:
            conv_node.input.append(bias_name)
        conv_node.output[0] = bn_node.output[0]

        model.graph.initializer.append(
            numpy_helper.from_array(folded_bias, name=bias_name)
        )

        self.delete_node_deferred(div_node)
        self.delete_node_deferred(bn_node)
