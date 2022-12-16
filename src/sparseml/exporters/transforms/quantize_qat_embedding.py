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

from onnx import ModelProto, helper, numpy_helper

from sparseml.exporters.transforms import OnnxTransform
from sparseml.exporters.transforms.utils import (
    INITIALIZER_MATCH,
    MatchResult,
    get_structural_matches,
    optional_node,
    quantize_array,
)
from sparseml.onnx.utils import ONNXGraph


_LOGGER = logging.getLogger(__name__)


class QuantizeQATEmbedding(OnnxTransform):
    """
    A transformation for quantizing qat embeddings.

    Transforms
    ```
    |       initializer
    |        |
    |        Q
    |        |
    | input  Dq
    |  |     |
    |  Gather
    |    |
    | optional Q
    |    |
    | optional Dq
    ```
    (where `Q` is QuantizeLinear, and `Dq` is Dequantize Linear)
    into
    ```
    | input
    |   |
    | Gather(UINT8 data initializer)
    |   |
    | DequantizeLinear
    ```
    """

    def transform(self, model: ModelProto) -> ModelProto:
        graph = ONNXGraph(model)
        matches = get_structural_matches(
            graph,
            op_type="Gather",
            parent_ops=[
                [],
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
            _LOGGER.debug(f"Matched {match}")
            self._transform_match(graph, model, match)
        _LOGGER.info(f"Converted {len(matches)} QAT embedding ops to UINT8")
        return model

    def _transform_match(self, graph: ONNXGraph, model: ModelProto, match: MatchResult):
        _, input_quant, input_dequant = match.parents[1]
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
        match.node.input[1] = embedding_quant_initializer.name

        if opt_output_quant is not None and opt_output_dequant is not None:
            # forward gather output to dequant input
            opt_output_dequant.input[0] = match.node.output[0]
            opt_output_dequant.input[1] = input_dequant.input[1]
            opt_output_dequant.input[2] = input_dequant.input[2]
            self.delete_node_deferred(opt_output_quant)
        else:
            # add new dequantize node after the match node
            match_node_quant_output = f"{match.node.output[0]}_quant"

            new_dequantize_node = helper.make_node(
                "DequantizeLinear",
                inputs=[
                    match_node_quant_output,
                    input_dequant.input[1],
                    input_dequant.input[2],
                ],
                outputs=[match.node.output[0]],
                name=f"dequantize_linear_{match.node.name}",
            )
            self.add_node_deferred(new_dequantize_node)
            match.node.output[0] = match_node_quant_output

        self.delete_node_deferred(input_quant)
        self.delete_node_deferred(input_dequant)
