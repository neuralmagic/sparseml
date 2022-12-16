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

from onnx import ModelProto, TensorProto

from sparseml.exporters.transforms import OnnxTransform
from sparseml.exporters.transforms.utils import (
    INITIALIZER_MATCH,
    MatchResult,
    add_quantized_conv_matmul_add_ops,
    get_quantization_params,
    get_structural_matches,
    optional_node,
)
from sparseml.onnx.utils import ONNXGraph


__all__ = ["MatMulToMatMulIntegerAddCastMul"]


_LOGGER = logging.getLogger(__name__)


class MatMulToMatMulIntegerAddCastMul(OnnxTransform):
    """
    A transform for converting a MatMul with kernel and bias into a
    quantized representation

    ```
    |     weight (initializer)
    |         |
    |         Q
    |         |
    | input   Dq
    |   |     |
    |   Dq   Transpose
    |     |   |
    |     MatMul  bias (initializer)
    |         |   |
    |         Add
    |         |
    |     optional Q
    |         |
    |     optional Dq
    ```
    (where `Q` is QuantizeLinear, and `Dq` is DequantizeLinear)
    into
    ```
    |   input
    |     |
    | MatMulInteger (with constant uint8 kernel)
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
        count = 0
        matches = get_structural_matches(
            graph,
            op_type="MatMul",
            parent_ops=[
                ["DequantizeLinear"],
                [
                    # weight should be initializer
                    INITIALIZER_MATCH,
                    "QuantizeLinear",
                    "DequantizeLinear",
                    "Transpose",
                ],
            ],
            children_ops=[
                [
                    "Add",
                    optional_node("QuantizeLinear"),
                    optional_node("DequantizeLinear"),
                ]
            ],
        )
        for match in matches:
            bias_init = graph.get_init_by_name(match.children[0][0].input[1])
            if bias_init is None:
                # bias initializer for add not present
                continue
            _LOGGER.debug(f"Matched {match}")
            self._transform_match(graph, model, match, bias_init)
            count += 1
        _LOGGER.info(f"Transformed {count} MatMul -> MatMulInteger")
        return model

    def _transform_match(
        self,
        graph: ONNXGraph,
        model: ModelProto,
        match: MatchResult,
        bias_init: TensorProto,
    ):
        matmul = match.node
        (input_dequant,) = match.parents[0]
        weight_init, weight_quant, weight_dequant, transpose = match.parents[1]
        add, opt_out_quant, opt_out_dequant = match.children[0]

        input_quantize_params = get_quantization_params(
            model, input_dequant, include_target=False
        )
        weight_quantize_params = get_quantization_params(
            model, weight_quant, include_target=True
        )
        # sanity check - matching handles this
        assert weight_quantize_params.target is not None

        add_quantized_conv_matmul_add_ops(
            model=model,
            node=matmul,
            input_quantize_node=input_dequant,
            weight_quantize_node=weight_quant,
            input_quantize_params=input_quantize_params,
            weight_quantize_params=weight_quantize_params,
            bias_initializer=bias_init,
            bias_add_name=add.name,
            target_output=(
                opt_out_dequant.output[0] if opt_out_dequant else add.output[0]
            ),
            transpose_weight=True,
            output_quantize_node=opt_out_quant,
            output_dequantize_node=opt_out_dequant,
        )

        # Clean up
        self.delete_node_deferred(weight_dequant)
        self.delete_node_deferred(weight_quant)
        self.delete_node_deferred(transpose)
        if len(graph.get_node_children(input_dequant)) == 1:
            self.delete_node_deferred(input_dequant)
        if opt_out_quant:
            self.delete_node_deferred(opt_out_quant)
        if opt_out_dequant:
            self.delete_node_deferred(opt_out_dequant)
        self.delete_node_deferred(matmul)
        self.delete_node_deferred(add)
