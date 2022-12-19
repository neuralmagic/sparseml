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

from onnx import ModelProto

from sparseml.exporters.transforms import OnnxTransform
from sparseml.exporters.transforms.utils import (
    INITIALIZER_MATCH,
    MatchResult,
    add_quantized_conv_matmul_add_ops,
    get_quantization_params,
    get_structural_matches,
)
from sparseml.onnx.utils import ONNXGraph


__all__ = ["ConvToConvIntegerAddCastMul"]


class ConvToConvIntegerAddCastMul(OnnxTransform):
    """
    A transform that attempts, if possible, to convert Convolution Op
    with kernel whose activations are not necessarily quantized into a
    ConvInteger followed by a bias add and cast to FP32.
    This MatMul is the result of quantizing native torch.matmul using QATMatMul

    Transforms:
    ```
    | input    weight (initializer)
    |   |      |
    |   Q      Q
    |   |      |
    |   Dq     Dq   bias (initializer)
    |     |    |    |
    |        Conv
    ```
    (where `Q` is QuantizeLinear, and `Dq` is DequantizeLinear)

    into

    ```
    |   input
    |     |
    | QuantizeLinear
    |     |
    | ConvInteger (with constant uint8 kernel)
    |     |
    | Add (constant bias + zero point correction)
    |     |
    | Cast (INT32 -> FP32)
    |     |
    | Mul (Rescale from bias scale)
    ```
    """

    def transform(self, model: ModelProto) -> ModelProto:
        graph = ONNXGraph(model)
        matches = get_structural_matches(
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
        )
        for match in matches:
            self.log_match(match)
            self._transform_match(graph, model, match)
        return model

    def _transform_match(self, graph: ONNXGraph, model: ModelProto, match: MatchResult):
        input_quant, input_dequant = match.parents[0]
        weight_init, weight_quantize_node, weight_dequantize_node = match.parents[1]
        (bias_init,) = match.parents[2]

        model = add_quantized_conv_matmul_add_ops(
            model=model,
            node=match.node,
            input_quantize_node=input_dequant,
            weight_quantize_node=weight_quantize_node,
            input_quantize_params=get_quantization_params(
                model, input_dequant, include_target=True
            ),
            weight_quantize_params=get_quantization_params(
                model, weight_quantize_node, include_target=True
            ),
            bias_initializer=bias_init,
            bias_add_name="{}_bias_add".format(match.node.name),
            target_output=match.node.output[0],
            transpose_weight=False,
        )

        # Clean up
        self.delete_node_deferred(weight_dequantize_node)
        self.delete_node_deferred(weight_quantize_node)
        if len(graph.get_node_children(input_dequant)) == 1:
            self.delete_node_deferred(input_dequant)
        self.delete_node_deferred(match.node)
