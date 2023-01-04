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

from typing import Any, Dict

from onnx import ModelProto

from sparseml.exporters.transforms import OnnxTransform
from sparseml.exporters.transforms.utils import (
    INITIALIZER_MATCH,
    MatchResult,
    add_quantized_conv_matmul_add_ops,
    any_of,
    get_quantization_params,
    get_structural_matches,
)
from sparseml.onnx.utils import ONNXGraph, get_node_attributes


__all__ = ["GemmToMatMulIntegerAddCastMul"]


class GemmToMatMulIntegerAddCastMul(OnnxTransform):
    """
    A transform for converting a Gemm op with kernel whose activations
    are not necessarily quantized into a MatMulInteger followed by
    a bias add and cast to FP32

    Transforms
    ```
    |     weight (intializer)
    |        |
    | input  Q
    |   |    |
    |  Q/Dq  Dq  bias (initializer)
    |     |  |  |
    |       Gemm
    ```
    (where `Q` is QuantizeLinear, and `Dq` is DequantizeLinear)

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
        matches = get_structural_matches(
            graph,
            op_type="Gemm",
            parent_ops=[
                [any_of("QuantizeLinear", "DequantizeLinear")],
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
        )
        for match in matches:
            attr = get_node_attributes(match.node)
            if (
                attr.get("alpha", 1.0) != 1.0
                or (attr.get("beta", 1.0) != 1.0)
                or attr.get("transA", False)
            ):
                # we do not currently handle Gemms with transposed A
                # or scalar multiples
                continue
            self.log_match(match)
            self._transform_match(graph, model, match, attr)

        return model

    def _transform_match(
        self,
        graph: ONNXGraph,
        model: ModelProto,
        match: MatchResult,
        gemm_attributes: Dict[str, Any],
    ):
        gemm = match.node
        (input_quant,) = match.parents[0]
        weight_init, weight_quant, weight_dequant = match.parents[1]
        (bias_init,) = match.parents[2]

        transpose_weight = bool(gemm_attributes.get("transB"))

        input_quantize_params = get_quantization_params(
            model, input_quant, include_target=False
        )
        weight_quantize_params = get_quantization_params(
            model, weight_quant, include_target=True
        )
        # sanity check - matching should handle this
        assert weight_quantize_params.target is not None

        add_quantized_conv_matmul_add_ops(
            model=model,
            node=match.node,
            input_quantize_node=input_quant,
            weight_quantize_node=weight_quant,
            input_quantize_params=input_quantize_params,
            weight_quantize_params=weight_quantize_params,
            bias_initializer=bias_init,
            bias_add_name=f"{gemm.name}_bias_add",
            target_output=gemm.output[0],
            transpose_weight=transpose_weight,
        )

        # Cleanup
        self.delete_node_deferred(weight_dequant)
        self.delete_node_deferred(weight_quant)
        if len(graph.get_node_children(input_quant)) == 1:
            self.delete_node_deferred(input_quant)
        self.delete_node_deferred(gemm)
