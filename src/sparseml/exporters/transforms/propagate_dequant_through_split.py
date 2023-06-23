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

import onnx
from onnx import ModelProto

from sparseml.exporters.transforms import OnnxTransform
from sparseml.exporters.transforms.utils import MatchResult, get_structural_matches
from sparseml.onnx.utils import ONNXGraph


__all__ = ["PropagateDequantThroughSplit"]


class PropagateDequantThroughSplit(OnnxTransform):
    """
    A pass for propagating DequantizeLinear nodes down through a split node
    so if there are quantized operations after the split they can
    be properly converted.
    Starting with:
    |         INPUT
    |              |
    |       DequantizeLinear
    |             |
    |           Split
    |         |   |   |
    Converts to:
    |                     INPUT
    |                         |
    |                       Split
    |                |         |           |
    | DequantizeLinear  DequantizeLinear  DequantizeLinear
    |         |                |                |
    """

    def transform(self, model: ModelProto) -> ModelProto:
        graph = ONNXGraph(model)
        matches = get_structural_matches(
            graph,
            parent_ops=[["DequantizeLinear"]],
            op_type="Split",
        )
        for match in matches:
            self.log_match(match)
            self._transform_match(model, match)
        return model

    def _transform_match(self, model: ModelProto, match: MatchResult):

        # Loop through the nodes that are children of the Split node
        # For every child, create a DequantizeLinear node and insert
        # between Split and child
        for split_output_id in range(len(match.node.output)):
            dequant_node_name = match.node.name + f"_dequant.{split_output_id}"
            dequant_node_output = match.node.output[split_output_id]
            dequant_node_input = dequant_node_name + "_input"

            # Input to DequantizeLinear node is the output of the Split node
            model.graph.node.append(
                onnx.helper.make_node(
                    "DequantizeLinear",
                    [
                        dequant_node_input,  # input
                        match.parents[0][0].input[1],  # scale
                        match.parents[0][0].input[2],  # zero point
                    ],
                    [dequant_node_output],
                    dequant_node_name,
                )
            )

            # Replace the output of the Split node with the input of
            # the new DequantizeLinear node
            match.node.output[split_output_id] = dequant_node_input

        # Set the input to the Split node to what was the input of the
        # original DequantizeLinear node
        match.node.input[0] = match.parents[0][0].input[0]

        # Remove original DequantizeLinear node
        self.delete_node_deferred(match.parents[0][0])
