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

import onnx
from onnx import ModelProto, TensorProto

from sparseml.exporters.transforms.kv_cache.transforms_base import (
    AdditionalTransformsBase,
)
from sparseml.onnx.utils.graph_editor import ONNXGraph
from sparseml.onnx.utils.helpers import get_nodes_by_input_id


_LOGGER = logging.getLogger(__name__)

__all__ = ["AdditionalTransformsCodeGen"]


class AdditionalTransformsCodeGen(AdditionalTransformsBase):

    # The patterns that match nodes that create
    # the `position_ids` and `causal_mask` tensors
    POSITION_IDS_MATCHING_PATTERN = dict(op_type="Range", children_ops=[["Unsqueeze"]])
    CAUSAL_MASK_MATCHING_PATTERN = dict(op_type="Slice", children_ops=[["Where"]])

    def transform(self, model: ModelProto) -> ModelProto:
        """
        1. Adds `positions` as an input to the model
        2. Adds `causal_mask` as an input to the model
        2. Finds the node that initially creates the `position_ids` tensor
        3. Updates the node to use the positions input instead of
           computing it from the Range op
        4. Finds the nodes that initially create the `causal_mask` tensors
        5. Updates the nodes to use the causal_mask input instead of
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

        model = self.inject_positions(model, position_ids_nodes, "Unsqueeze")

        causal_mask_nodes = self.find_nodes_by_pattern(
            model, pattern=self.CAUSAL_MASK_MATCHING_PATTERN
        )
        model = self.inject_causal_mask(model, causal_mask_nodes, "Where")
        model = self.adjust_causal_mask(model)
        return model

    def adjust_causal_mask(self, model: ModelProto) -> ModelProto:
        """
        Insert a `Cast` node after the causal mask input to change
        the initial int64, to a mask of bools expected by the model.

        Transform:
        ```
        |       causal_mask
        |            |
        |   causal_mask_input_child
        ```
        to:
        ```
        |       causal_mask
        |            |
        |          Cast
        |        (to bool)
        |            |
        |            |
        |   causal_mask_input_child

        The resulting node will change the input boolean mask
        e.g.
        ```
        causal_mask =
            [[[[1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1]]]]
        ```

        to a mask of bools:
        ```
        causal_mask_adjusted =
            [[[[True, True, True, False, False, False],
                [True, True, True, True, False, False],
                [True, True, True, True, True, False],
                [True, True, True, True, True, True]]]]
        ```

        :param model: the model to update
        :return: the updated model
        """

        graph = ONNXGraph(model)

        cast_node_output = f"{self.CAUSAL_MASK_NAME}_adjusted"

        cast_node = onnx.helper.make_node(
            "Cast",
            inputs=[self.CAUSAL_MASK_NAME],
            outputs=[cast_node_output],
            to=TensorProto.BOOL,
        )

        # get the nodes that take the causal mask as input
        # and replace the input with the adjusted causal mask input
        causal_mask_input_children = get_nodes_by_input_id(model, self.CAUSAL_MASK_NAME)
        for causal_mask_input_child in causal_mask_input_children:
            for idx, input_name in enumerate(causal_mask_input_child.input):
                if input_name == self.CAUSAL_MASK_NAME:
                    causal_mask_input_child.input[idx] = cast_node_output

        graph.add_node(cast_node)
        self.log_match(cast_node)

        _LOGGER.info(f"Successfully adjusted the {self.CAUSAL_MASK_NAME} input")

        return model
