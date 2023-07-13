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
from sparseml.onnx.utils import ONNXGraph


__all__ = ["PositionsAdjustmentOPT"]


# name position embeddings weights
_EMBED_POSITIONS_ID = "model.decoder.embed_positions.weight"


class PositionsAdjustmentOPT(PositionsAdjustmentBase):
    """
    Base class for model architecture specific transforms to adjust graph
    to take input_id positions as an argument rather than computing them
    based on input. This provides a better source of truth rather than
    computing in a static graph where factors such as number of tokens,
    cache size, and padding may affect accurate, efficient static
    computation of the position indices.

    Positions should be the same shape as input_ids where each value
    is the corresponding integer position of the token id in the overall
    sequence. Padding tokens are not counted towards positions and
    should be inputted as 0.

    When running a model for a single input id with `n` previously
    processed tokens (prompt seq len + number of tokens generated already)

    This transform will replace the input to the position embeddings gather
    by an explicit onnx graph input.  Will delete any operations
    used to compute the positions that are now no longer used in the
    graph. Optionally keeps an offset `Add` node that is unique to the
    OPT graph.

    Transforms
    ```
    |  Graph computed positions
    |     |
    |   Add (Optional)
    |     |
    |  Gather(model.decoder.embed_positions.weight)
    ```
    Into
    ```
    |  Explicit Graph input  (deletes now orphaned nodes to compute positions)
    |     |
    |   Add (Optional)
    |     |
    |  Gather(model.decoder.embed_positions.weight)
    ```

    """

    def transform(self, model: ModelProto) -> ModelProto:
        model = self.add_positions_input(model)
        position_embeddings_node = self.find_embed_positions_gather_node(model)
        model = self._update_position_embeddings_for_graph_input(
            model, position_embeddings_node
        )
        return model

    @classmethod
    def find_embed_positions_gather_node(cls, model: ModelProto) -> NodeProto:
        for node in model.graph.node:
            if node.op_type != "Gather":
                continue
            if node.input[0] == _EMBED_POSITIONS_ID:
                # found the embed_positions_gather_node
                return node
        raise RuntimeError(
            f"Unable to find position embeddings gather node with id "
            f"{_EMBED_POSITIONS_ID} in {cls.__name__}"
        )

    def _update_position_embeddings_for_graph_input(
        self, model: ModelProto, position_embeddings_node: NodeProto
    ) -> ModelProto:
        graph = ONNXGraph(model)

        # select target node to update as positions input
        position_embeddings_parent = graph.get_node_single_parent(
            position_embeddings_node, index=1
        )

        if not isinstance(position_embeddings_parent, NodeProto):
            raise RuntimeError(
                f"Unable to find input to position embeddings node: "
                f"{position_embeddings_node.name} as a node in the given model"
            )

        if position_embeddings_parent.op_type == "Add":
            # OPT has a special Add offset for position ids, allow this
            # to be where positions are fed instead
            target_update_node = position_embeddings_parent
            target_input_idx = 0  # assume positions are first input to the Add
        else:
            target_update_node = position_embeddings_node
            target_input_idx = 1  # gather idxs

        # reroute target node input to the positions graph input
        old_positions_input = target_update_node.input[target_input_idx]
        target_update_node.input[target_input_idx] = self.POSITIONS_NAME
        graph.update()
        self.log_match(target_update_node)

        nodes_to_delete = graph.find_orphaned_nodes(
            graph.get_node_by_output_id(old_positions_input)
        )
        [self.log_match(node) for node in nodes_to_delete]

        graph.delete_nodes(nodes_to_delete)
        graph.update()
        graph.delete_unused_initializers()

        return model
