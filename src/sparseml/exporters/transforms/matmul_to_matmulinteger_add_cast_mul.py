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
    delete_quant_node,
    get_quantization_params,
    get_structural_matches,
    optional_node,
)
from sparseml.onnx.utils import (
    ONNXGraph,
    get_init_by_name,
    remove_node_and_params_from_graph,
)


_LOGGER = logging.getLogger(__name__)

__all__ = ["MatMulToMatMulIntegerAddCastMul"]


class MatMulToMatMulIntegerAddCastMul(OnnxTransform):
    """
    A transform for converting a MatMul with kernel and bias into a
    quantized representation

    ```
    | Starting with:
    |                        weight (initializer)
    |                            |
    |                      QuantizeLinear
    |                            |
    |          INPUT     DequantizeLinear
    |            |               |
    |     DequantizeLinear   Transpose
    |                  |      |
    |                   MatMul    bias (initializer)
    |                         |   |
    |                          Add
    |                          |
    |                      QuantizeLinear (Optional)
    |                          |
    |                      DequantizeLinear (Optional)
    |                          |
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
            bias_init = get_init_by_name(model, match.children[0][0].input[1])
            if bias_init is None:
                # bias initializer not present
                continue

            _LOGGER.debug(f"Found structural match {match.node.name}")
            self._do_transform(model, match, bias_init)

        graph = ONNXGraph(model)
        graph.delete_unused_initializers()
        graph.sort_nodes_topologically()
        return model

    def _do_transform(
        self, model: ModelProto, match: MatchResult, bias_init: TensorProto
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

        # delete folded quantization ops
        delete_quant_node(model, weight_dequant)
        delete_quant_node(model, weight_quant)
        remove_node_and_params_from_graph(model, transpose)

        # only delete input node if the matmul is the only child
        current_graph = ONNXGraph(model)
        if len(current_graph.get_node_children(input_dequant)) == 1:
            delete_quant_node(model, input_dequant)
        if opt_out_quant:
            delete_quant_node(model, opt_out_quant)
        if opt_out_dequant:
            delete_quant_node(model, opt_out_dequant)

        # delete original Gemm node
        remove_node_and_params_from_graph(model, matmul)
        # delete original Add node
        remove_node_and_params_from_graph(model, add)
