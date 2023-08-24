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

import numpy
from onnx import ModelProto, numpy_helper

from sparseml.exporters.transforms.kv_cache.transforms_base import (
    AdditionalTransformsBase,
)
from sparseml.onnx.utils.graph_editor import ONNXGraph


__all__ = ["AdditionalTransformsLLAMA"]

_LOGGER = logging.getLogger(__name__)


class AdditionalTransformsLLAMA(AdditionalTransformsBase):

    POSITION_IDS_MATCHING_PATTERN = dict(op_type="Range", children_ops=[["Unsqueeze"]])
    CAUSAL_MASK_MATCHING_PATTERN = dict(op_type="Expand", children_ops=[["Add"]])

    def transform(self, model: ModelProto) -> ModelProto:
        """
        1  Updates the Slice nodes in the attention heads by extending the `ends`
        operator
        2. Adds `positions` as an input to the model
        3. Adds `causal_mask` as an input to the model
        4. Finds the node that initially creates the `position_ids` tensor
        5. Updates the node to use the positions input instead of
           computing it from the Range op
        6. Finds the nodes that initially create the `causal_mask` tensors
        7. Updates the nodes to use the causal_mask input instead of
              computing it from the Expand op
        8. Update the masks to be floats, as expected by the model

        :param model: model to update
        :return: updated model
        """

        model = self.update_slice_nodes_for_positions_input(model)
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

        model = self.inject_positions(model, position_ids_nodes, "Unsqueeze")

        causal_mask_nodes = self.find_nodes_by_pattern(
            model, pattern=self.CAUSAL_MASK_MATCHING_PATTERN
        )
        model = self.inject_causal_mask(model, causal_mask_nodes, "Add")
        model = self.adjust_causal_mask(model)
        return model

    def update_slice_nodes_for_positions_input(self, model: ModelProto) -> ModelProto:
        """
        Update the Slice nodes in the attention heads such that the `ends` operator is
        set to the max int value. This value is missing from the export and is required
        for the position ids injection. This is because the onnx export limits access to
        the entire sin_cached and cos_cached tables, which results in an index error
        with the position ids:

        https://github.com/huggingface/transformers/blob/
        7a6efe1e9f756f585f2ffe5ada22cf6b15edd23b/src/transformers/models/llama/
        modeling_llama.py#L180.

        By updating the `ends` operator, access is allowed to the entire tables.
        The Slice nodes are identified based on if they contain the `data` operator
        as an input, which have the name `onnx::Slice_...`. Nodes with this name have
        their `ends` operator updated to point to a 1x1 tensor containing the max
        int value.

        :param model: model to update
        :return: updated model with Slice nodes in the attention heads updated
        """
        SLICE_MAX_INT_NAME = "slice_max_int"
        arr = numpy.array(numpy.iinfo(numpy.intp).max).reshape(
            1,
        )
        max_int_tensor = numpy_helper.from_array(arr, name=SLICE_MAX_INT_NAME)

        nodes_found = 0
        for node in model.graph.node:
            if node.op_type == "Slice":
                data = node.input[0]
                if "onnx::" in data:
                    node.input[2] = SLICE_MAX_INT_NAME
                    nodes_found += 1
                    self.log_match(node)

        _LOGGER.info(f"Found {nodes_found} slice nodes to update")

        model.graph.initializer.append(max_int_tensor)
        ONNXGraph(model).delete_orphaned_node_branches()
        return model
