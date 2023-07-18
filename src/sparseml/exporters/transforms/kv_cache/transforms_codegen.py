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

from typing import List

from onnx import ModelProto, NodeProto

from sparseml.exporters.transforms.kv_cache.transforms_base import (
    AdditionalTransformsBase,
)
from sparseml.onnx.utils import ONNXGraph


__all__ = ["AdditionalTransformsCodeGen"]


class AdditionalTransformsCodeGen(AdditionalTransformsBase):

    # The patterns that match nodes that create
    # the `position_ids` and `causal_mask` tensors
    POSITION_IDS_MATCHING_PATTERN = dict(op_type="Range")
    CAUSAL_MASK_MATCHING_PATTERN = dict(op_type="Slice", children_ops=[["Where"]])

    def transform(self, model: ModelProto) -> ModelProto:
        """
        1. Adds `positions` as an input to the model
        2. Adds `causal_mask` as an input to the model
        2. Finds the node that initially creates the `position_ids` tensor
        3. Updates the node to use the positions input instead of
           computing it from the Range op
        4. Finds the node that initially creates the `causal_mask` tensor
        5. Updates the node to use the causal_mask input instead of
              computing it from the Slice op

        :param model: model to update
        :return: updated model
        """
        model = self.add_positions_input(model)
        model = self.add_causal_mask_input(model)

        position_ids_nodes = self.find_nodes_by_pattern(
            model, pattern=self.POSITION_IDS_MATCHING_PATTERN
        )
        if len(position_ids_nodes) != 1:
            raise ValueError(
                "Expected to find exactly one node matching "
                f"the pattern {self.POSITION_IDS_MATCHING_PATTERN}, "
                f"found {len(position_ids_nodes)}"
            )
        position_ids_node = position_ids_nodes[0]
        model = self._update_position_embeddings_for_graph_input(
            model, position_ids_node
        )

        causal_mask_nodes = self.find_nodes_by_pattern(
            model, pattern=self.CAUSAL_MASK_MATCHING_PATTERN
        )
        model = self._update_causal_mask_for_graph_input(model, causal_mask_nodes)
        return model

    def _update_causal_mask_for_graph_input(
        self, model: ModelProto, causal_mask_nodes: List[NodeProto]
    ) -> ModelProto:

        graph = ONNXGraph(model)
        orphaned_nodes = []
        for node in causal_mask_nodes:
            child_node = graph.get_node_children(node)[0]
            # child node is the `Where` node
            assert (
                child_node.op_type == "Where"
            ), f"Expected to find `Where` node, found {child_node.op_type}"
            output_to_replace = node.output[0]
            self.log_match(node)
            for idx, input_name in enumerate(child_node.input):
                if input_name == output_to_replace:
                    graph.update_node_input(child_node, self.CAUSAL_MASK_NAME, idx)

            orphaned_nodes.extend(graph.find_orphaned_nodes(node))

        graph.delete_nodes(orphaned_nodes)
        graph.update()
        graph.delete_unused_initializers()

        return model

    def _update_position_embeddings_for_graph_input(
        self, model: ModelProto, position_embeddings_ids_node: NodeProto
    ) -> ModelProto:

        graph = ONNXGraph(model)
        node = position_embeddings_ids_node
        child_node = graph.get_node_children(node)[0]
        # child_node is the `Unsqueeze` node
        assert (
            child_node.op_type == "Unsqueeze"
        ), f"Expected to find `Unsqueeze` node, found {child_node.op_type}"
        output_to_replace = node.output[0]
        self.log_match(node)
        for idx, input_name in enumerate(child_node.input):
            if input_name == output_to_replace:
                graph.update_node_input(child_node, self.POSITIONS_NAME, idx)

        orphaned_nodes = graph.find_orphaned_nodes(node)
        [self.log_match(node) for node in orphaned_nodes]
        graph.delete_nodes(orphaned_nodes)
        graph.update()
        graph.delete_unused_initializers()

        return model
