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

from onnx import ModelProto, numpy_helper

from sparseml.exporters.transforms import OnnxTransform
from sparseml.exporters.transforms.utils import (
    INITIALIZER_MATCH,
    MatchResult,
    get_structural_matches,
    optional_node,
    quantize_array,
)
from sparseml.onnx.utils import ONNXGraph


class QuantizeQATEmbedding(OnnxTransform):
    """
    A transformation for quantizing qat embeddings.

    Transforms
    ```
    | initializer
    | |
    | Q
    | |
    | Dq    input
    | |     |
    |  Gather
    |    |
    | optional Q
    |    |
    | optional Dq
    ```
    (where `Q` is QuantizeLinear, and `Dq` is Dequantize Linear)
    into
    ```
    | uint8 init    input
    |     |      |
    |      Gather
    |        |
    |        Dq
    ```
    """

    def transform(self, model: ModelProto) -> ModelProto:
        graph = ONNXGraph(model)
        matches = get_structural_matches(
            graph,
            op_type="Gather",
            parent_ops=[
                [
                    INITIALIZER_MATCH,
                    "QuantizeLinear",
                    "DequantizeLinear",
                ],
            ],
            children_ops=[
                [optional_node("QuantizeLinear"), optional_node("DequantizeLinear")]
            ],
        )
        for match in matches:
            # check that all input_quant's inputs are initializers
            for init_name in match.parents[0][1].input[:3]:
                if graph.get_init_by_name(init_name) is None:
                    continue

            self.log_match(match)
            self._transform_match(graph, model, match)
        return model

    def _transform_match(self, graph: ONNXGraph, model: ModelProto, match: MatchResult):
        _, input_quant, input_dequant = match.parents[0]
        opt_output_quant, opt_output_dequant = match.children[0]

        # quantize embedding
        embedding_initializer = graph.get_init_by_name(input_quant.input[0])
        scale_initializer = graph.get_init_by_name(input_quant.input[1])
        zero_point_initializer = graph.get_init_by_name(input_quant.input[2])

        # arrays from embedding initializer
        embedding = numpy_helper.to_array(embedding_initializer)
        scale = numpy_helper.to_array(scale_initializer)
        zero_point = numpy_helper.to_array(zero_point_initializer)

        embedding_quant = quantize_array(embedding, scale, zero_point, zero_point.dtype)
        embedding_quant_initializer = numpy_helper.from_array(
            embedding_quant, name=f"{embedding_initializer.name}_quant"
        )

        # update graph
        model.graph.initializer.append(embedding_quant_initializer)
        match.node.input[0] = embedding_quant_initializer.name

        has_qdq = opt_output_quant is not None and opt_output_dequant is not None
        if (
            has_qdq
            # NOTE: we only support this branch for qdq with 1 child
            and len(graph.get_node_children(match.node)) == 1
            and len(graph.get_node_children(opt_output_quant)) == 1
        ):
            # forward gather output to dequant input
            opt_output_dequant.input[0] = match.node.output[0]
            opt_output_dequant.input[1] = input_dequant.input[1]
            opt_output_dequant.input[2] = input_dequant.input[2]
            self.delete_node_deferred(input_quant)
            self.delete_node_deferred(input_dequant)
            self.delete_node_deferred(opt_output_quant)
        else:
            # use input dequant to dequantize output
            embedding_quant_output_id = f"{match.node.output[0]}_quant"
            input_dequant.input[0] = embedding_quant_output_id
            input_dequant.output[0] = match.node.output[0]
            match.node.output[0] = embedding_quant_output_id
            self.delete_node_deferred(input_quant)
