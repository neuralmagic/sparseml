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

from onnx import ModelProto, TensorProto

from sparseml.exporters.transforms import OnnxTransform
from sparseml.exporters.transforms.utils import (
    INITIALIZER_MATCH,
    MatchResult,
    add_quantized_conv_matmul_add_ops,
    any_of,
    get_quantization_params,
    get_structural_matches,
    optional_node,
)
from sparseml.onnx.utils import ONNXGraph


__all__ = ["MatMulAddToMatMulIntegerAddCastMul"]


class MatMulAddToMatMulIntegerAddCastMul(OnnxTransform):
    """
    A transform for converting a MatMul with kernel and bias into a
    quantized representation

    If add or bias initializer does not exist, the bias is skipped

    ```
    |     weight (initializer)
    |         |
    |         Q
    |         |
    | input   Dq
    |   |     |
    |  Q/Dq   optional Transpose
    |     |   |
    |     MatMul  bias (initializer) (optional)
    |         |   |
    |         Add (optional)
    ```
    (where `Q` is QuantizeLinear, and `Dq` is DequantizeLinear)
    into
    ```
    |   input
    |     |
    | MatMulInteger (with constant uint8 kernel)
    |     |
    | Add (constant bias + zero point correction) (optional)
    |     |
    | Cast (INT32 -> FP32)
    |     |
    | Mul (Rescale from bias scale)
    ```
    """

    def transform(self, model: ModelProto) -> ModelProto:
        graph = ONNXGraph(model)

        # Weight on input 0
        matches = get_structural_matches(
            graph,
            op_type="MatMul",
            parent_ops=[
                [
                    # weight should be initializer
                    INITIALIZER_MATCH,
                    "QuantizeLinear",
                    "DequantizeLinear",
                    optional_node("Transpose"),
                ],
                [any_of("QuantizeLinear", "DequantizeLinear")],
            ],
            children_ops=[[optional_node("Add")]],
        )
        for match in matches:
            add_node = match.children[0][0]
            bias_init = None
            if add_node:
                # NOTE: bias could be either input 0 or 1 of add node
                # if add does not have a bias initializer,
                # still do conversion, but do not fold the bias add to rescale
                bias_init = graph.get_init_by_name(match.children[0][0].input[1])
                if bias_init is None:
                    bias_init = graph.get_init_by_name(match.children[0][0].input[0])
            self.log_match(match)
            self._transform_match(graph, model, match, bias_init, 0)

        # Weight on input 1
        matches = get_structural_matches(
            graph,
            op_type="MatMul",
            parent_ops=[
                [any_of("QuantizeLinear", "DequantizeLinear")],
                [
                    # weight should be initializer
                    INITIALIZER_MATCH,
                    "QuantizeLinear",
                    "DequantizeLinear",
                    optional_node("Transpose"),
                ],
            ],
            children_ops=[[optional_node("Add")]],
        )
        for match in matches:
            add_node = match.children[0][0]
            bias_init = None
            if add_node:
                # NOTE: bias could be either input 0 or 1 of add node
                # if add does not have a bias initializer,
                # still do conversion, but do not fold the bias add to rescale
                bias_init = graph.get_init_by_name(match.children[0][0].input[1])
                if bias_init is None:
                    bias_init = graph.get_init_by_name(match.children[0][0].input[0])
            self.log_match(match)
            self._transform_match(graph, model, match, bias_init, 1)

        return model

    def _transform_match(
        self,
        graph: ONNXGraph,
        model: ModelProto,
        match: MatchResult,
        bias_init: TensorProto,
        weight_parent: int,
    ):
        matmul = match.node
        if weight_parent == 0:
            (input_quant,) = match.parents[1]
            weight_init, weight_quant, weight_dequant, opt_transpose = match.parents[0]
        else:
            (input_quant,) = match.parents[0]
            weight_init, weight_quant, weight_dequant, opt_transpose = match.parents[1]
        (add,) = match.children[0]

        input_quantize_params = get_quantization_params(
            model, input_quant, include_target=False
        )
        weight_quantize_params = get_quantization_params(
            model, weight_quant, include_target=True
        )
        # sanity check - matching handles this
        assert weight_quantize_params.target is not None

        add_quantized_conv_matmul_add_ops(
            model=model,
            node=matmul,
            input_quantize_node=input_quant,
            weight_quantize_node=weight_quant,
            input_quantize_params=input_quantize_params,
            weight_quantize_params=weight_quantize_params,
            bias_initializer=bias_init,
            bias_add_name=add.name if add else None,
            target_output=add.output[0] if add and bias_init else None,
            transpose_weight=opt_transpose is not None,
        )

        # Clean up
        self.delete_node_deferred(weight_dequant)
        self.delete_node_deferred(weight_quant)
        if opt_transpose is not None:
            self.delete_node_deferred(opt_transpose)
        if len(graph.get_node_children(input_quant)) == 1:
            self.delete_node_deferred(input_quant)
        self.delete_node_deferred(matmul)
        if bias_init is not None:
            # add converted to quantized - delete previous add node
            self.delete_node_deferred(add)
