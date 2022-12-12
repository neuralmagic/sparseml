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
from typing import Any, Dict

from onnx import ModelProto

from sparseml.exporters.transforms import OnnxTransform
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
    get_node_attributes,
    remove_node_and_params_from_graph,
)


_LOGGER = logging.getLogger(__name__)

__all__ = ["GemmToMatMulIntegerAddCastMul"]


class GemmToMatMulIntegerAddCastMul(OnnxTransform):
    """
    A transform for converting a Gemm op with kernel whose activations
    are not necessarily quantized into a MatMulInteger followed by
    a bias add and cast to FP32

    ```
    | Starting with:
    |
    |                       weight (initializer)
    |                         |
    |          INPUT        QuantizeLinear
    |            |            |
    |     DequantizeLinear  DequantizeLinear   bias (initializer)
    |                  |      |              /
    |                       Gemm
    |                         |
    |                       OUTPUT
    | We end up converting to:
    |       INPUT
    |         |
    |     MatMulInteger (with constant uint8 kernel)
    |         |
    |     Add (constant bias + zero point correction)
    |         |
    |     Cast (INT32 -> FP32)
    |         |
    |     Mul (Rescale from bias scale)
    |         |
    |       OUTPUT
    ```
    """

    def transform(self, model: ModelProto) -> ModelProto:
        matches = get_structural_matches(
            ONNXGraph(model),
            op_type="Gemm",
            parent_ops=[
                ["DequantizeLinear"],
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
            _LOGGER.debug(f"Found structural match {match}")
            attr = get_node_attributes(match.node)
            if (
                attr.get("alpha", 1.0) != 1.0
                or (attr.get("beta", 1.0) != 1.0)
                or attr.get("transA", False)
            ):
                # we do not currently handle Gemms with transposed A
                # or scalar multiples
                _LOGGER.debug(f"Skipping due to unsupported attributes {attr}")
                continue

            self._do_transform(model, match, attr)

        return model

    def _do_transform(
        self, model: ModelProto, match: MatchResult, gemm_attributes: Dict[str, Any]
    ):
        gemm = match.node
        (input_dequant,) = match.parents[0]
        weight_init, weight_quant, weight_dequant = match.parents[1]
        (bias_init,) = match.parents[2]

        transpose_weight = bool(gemm_attributes.get("transB"))

        input_quantize_params = get_quantization_params(
            model, input_dequant, include_target=False
        )
        weight_quantize_params = get_quantization_params(
            model, weight_quant, include_target=True
        )
        # sanity check - matching should handle this
        assert weight_quantize_params.target is not None

        add_quantized_conv_matmul_add_ops(
            model=model,
            node=match.node,
            input_quantize_node=input_dequant,
            weight_quantize_node=weight_quant,
            input_quantize_params=input_quantize_params,
            weight_quantize_params=weight_quantize_params,
            bias_initializer=bias_init,
            bias_add_name=f"{gemm.name}_bias_add",
            target_output=gemm.output[0],
            transpose_weight=transpose_weight,
        )

        # Cleanup
        # delete folded quantization ops
        delete_quant_node(model, weight_dequant)
        delete_quant_node(model, weight_quant)

        # only delete input node if the matmul is the only child
        current_graph = ONNXGraph(model)
        if len(current_graph.get_node_children(input_dequant)) == 1:
            delete_quant_node(model, input_dequant)

        # delete original Gemm node
        remove_node_and_params_from_graph(model, gemm)
