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

from onnx import ModelProto, NodeProto

from sparseml.exporters.transforms.kv_cache.positions_adjustment_base import (
    PositionsAdjustmentBase,
)
from sparseml.exporters.transforms.utils.matching import get_structural_matches
from sparseml.onnx.utils import ONNXGraph


__all__ = ["PositionsAdjustmentCodeGen"]


class PositionsAdjustmentCodeGen(PositionsAdjustmentBase):

    # The pattern that matches the node that creates
    # the `position_ids` tensor
    POSITION_IDS_MATCHING_PATTERN = dict(op_type="Range")

    def transform(self, model: ModelProto) -> ModelProto:
        """
        1. Adds `positions` as an input to the model
        2. Finds the node that initially creates the `position_ids` tensor
        3. Updates the node to use the positions input instead of
           computing it from the Range op

        :param model: model to update
        :return: updated model
        """
        model = self.add_positions_input(model)
        position_ids_node = self.find_position_ids_range_node(model)
        model = self._update_position_embeddings_for_graph_input(
            model, position_ids_node
        )
        return model

    def find_position_ids_range_node(self, model: ModelProto) -> NodeProto:
        """
        Find the node that creates the `position_ids` tensor
        :param model: the ONNX model
        :return: the node that creates the `position_ids` tensor
        """
        graph = ONNXGraph(model)
        position_ids_node = get_structural_matches(
            graph, **self.POSITION_IDS_MATCHING_PATTERN
        )
        if len(position_ids_node) != 1:
            raise ValueError(
                f"Expected to find 1 position node, found {len(position_ids_node)}"
            )
        return position_ids_node[0].node

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
        self.log_match(child_node)
        for idx, input_name in enumerate(child_node.input):
            if input_name == output_to_replace:
                graph.update_node_input(child_node, self.POSITIONS_NAME, idx)

        orphaned_nodes = graph.find_orphaned_nodes(node)
        [self.log_match(node) for node in orphaned_nodes]
        graph.delete_nodes(orphaned_nodes)
        graph.update()
        graph.delete_unused_initializers()

        return model
