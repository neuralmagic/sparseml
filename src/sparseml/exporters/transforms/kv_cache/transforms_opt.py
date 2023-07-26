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

import numpy
import onnx
from onnx import ModelProto

from sparseml.exporters.transforms.kv_cache.transforms_base import (
    AdditionalTransformsBase,
)
from sparseml.onnx.utils.graph_editor import ONNXGraph
from sparseml.onnx.utils.helpers import get_nodes_by_input_id


__all__ = ["AdditionalTransformsOPT"]


class AdditionalTransformsOPT(AdditionalTransformsBase):
    """
    1. Adds `positions` as an input to the model
    2. Adds `causal_mask` as an input to the model
    2. Finds the node that initially creates the `embed_position.weight` tensor
    3. Updates the node to use the positions input instead of
       computing it from the Sub op
    4. Finds the nodes that initially create the `causal_mask` tensors
    5. Updates the nodes to use the causal_mask input instead of
          computing it from the Expand op

    :param model: model to update
    :return: updated model
    """

    POSITION_EMBEDDINGS_IDS_MATCHING_PATTERN = dict(
        op_type="Sub",
        children_ops=[["Add"]],
    )
    CAUSAL_MASK_MATCHING_PATTERN = dict(op_type="Expand", children_ops=[["Add"]])

    def transform(self, model: ModelProto) -> ModelProto:
        model = self.add_positions_input(model)
        model = self.add_causal_mask_input(model)

        position_embeddings_nodes = self.find_nodes_by_pattern(
            model, pattern=self.POSITION_EMBEDDINGS_IDS_MATCHING_PATTERN
        )
        if len(position_embeddings_nodes) != 1:
            raise ValueError(
                "Expected to find exactly one node matching "
                f"the pattern {self.POSITION_EMBEDDINGS_IDS_MATCHING_PATTERN}, "
                f"found {len(position_embeddings_nodes)}"
            )
        model = self.inject_positions(model, position_embeddings_nodes, "Add")

        causal_mask_nodes = self.find_nodes_by_pattern(
            model, pattern=self.CAUSAL_MASK_MATCHING_PATTERN
        )
        model = self.inject_causal_mask(model, causal_mask_nodes, "Add")
        model = self.adjust_causal_mask(model)
        return model

    def adjust_causal_mask(self, model: ModelProto) -> ModelProto:
        """
        Insert a `Where`` node after the causal mask input to change
        the initial boolean mask, to a mask of floats expected by the model.


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
        |          Where
        |            |
        |   causal_mask_input_child

        The resulting node will change the input boolean mask
        e.g.
        ```
        causal_mask =
            [[[[True, True, True, False, False, False],
               [True, True, True, True, False, False],
               [True, True, True, True, True, False],
               [True, True, True, True, True, True]]]]
        ```

        to a mask of floats:

        ```
        x = numpy.finfo(numpy.float32).min
        causal_mask_adjusted =
            [[[[0.0, 0.0, 0.0, x, x, x],
               [0.0, 0.0, 0.0, 0.0, x, x],
               [0.0, 0.0, 0.0, 0.0, 0.0, x],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]]
        ```

        :param model: the model to update
        :return: the updated model
        """

        graph = ONNXGraph(model)

        condition_true = 0.0
        condition_false = numpy.finfo(numpy.float32).min

        condition_true_initializer = onnx.helper.make_tensor(
            name="condition_true",
            data_type=onnx.TensorProto.FLOAT,
            # TODO: how to set proper dims?
            #  This works, but is kinda ugly
            dims=[1, 1, 1, 1],
            vals=[condition_true],
        )

        condition_false_initializer = onnx.helper.make_tensor(
            name="condition_false",
            data_type=onnx.TensorProto.FLOAT,
            dims=[1, 1, 1, 1],
            vals=[condition_false],
        )

        where_node_output = f"{self.CAUSAL_MASK_NAME} adjusted"

        where_node = onnx.helper.make_node(
            "Where",
            inputs=[self.CAUSAL_MASK_NAME, "condition_true", "condition_false"],
            outputs=[where_node_output],
        )

        graph.add_node(where_node)
        model.graph.initializer.extend(
            [condition_false_initializer, condition_true_initializer]
        )

        # get the node that takes the causal mask as input
        # and replace the input with the adjusted causal mask input
        causal_mask_input_child = get_nodes_by_input_id(model, self.CAUSAL_MASK_NAME)[0]

        for idx, input_name in enumerate(causal_mask_input_child.input):
            if input_name == self.CAUSAL_MASK_NAME:
                causal_mask_input_child.input[idx] = where_node_output

        return model
