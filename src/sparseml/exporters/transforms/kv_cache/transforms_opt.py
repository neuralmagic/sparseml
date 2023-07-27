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
import onnx
from onnx import ModelProto

from sparseml.exporters.transforms.kv_cache.transforms_base import (
    AdditionalTransformsBase,
)
from sparseml.onnx.utils.graph_editor import ONNXGraph
from sparseml.onnx.utils.helpers import get_nodes_by_input_id


__all__ = ["AdditionalTransformsOPT"]

_LOGGER = logging.getLogger(__name__)


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
        Insert a `Cast`, `Sub` and `Mul` nodes after the causal mask input to change
        the initial int64, to a mask of floats expected by the model.

        Transform:
        ```
        |       causal_mask
        |            |
        |   causal_mask_input_child
        ```
        to:
        ```
        |       causal_mask (1 and 0)
        |            |
        |          Cast  (output -> 1.0 and 0.0)
        |            |
        |           Sub (output -> 0.0 and -1.0)
        |            |
        |           Mul (output -> 0.0 and numpy.finfo(numpy.float32).min)
        |            |
        |   causal_mask_input_child

        The resulting node will change the input int64 mask
        e.g.
        ```
        causal_mask =
            [[[[1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1]]]]
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

        ones_initializer = onnx.helper.make_tensor(
            name="ones_initializer",
            data_type=onnx.TensorProto.FLOAT,
            dims=[1],
            vals=[1.0],
        )

        floating_point_limit_initializer = onnx.helper.make_tensor(
            name="floating_point_limit_initializer",
            data_type=onnx.TensorProto.FLOAT,
            dims=[1],
            vals=[-numpy.finfo(numpy.float32).min],
        )

        cast_node = onnx.helper.make_node(
            "Cast",
            inputs=[self.CAUSAL_MASK_NAME],
            outputs=[f"{self.CAUSAL_MASK_NAME}_cast"],
            to=onnx.TensorProto.FLOAT,
        )

        sub_node = onnx.helper.make_node(
            "Sub",
            inputs=[f"{self.CAUSAL_MASK_NAME}_cast", ones_initializer.name],
            outputs=[f"{self.CAUSAL_MASK_NAME}_sub"],
        )

        mul_node = onnx.helper.make_node(
            "Mul",
            inputs=[
                f"{self.CAUSAL_MASK_NAME}_sub",
                floating_point_limit_initializer.name,
            ],
            outputs=[f"{self.CAUSAL_MASK_NAME}_mul"],
        )

        new_nodes = [cast_node, sub_node, mul_node]

        # get the node that takes the causal mask as input
        # and replace the input with the adjusted causal mask input
        causal_mask_input_child = get_nodes_by_input_id(model, self.CAUSAL_MASK_NAME)[0]

        for idx, input_name in enumerate(causal_mask_input_child.input):
            if input_name == self.CAUSAL_MASK_NAME:
                causal_mask_input_child.input[idx] = f"{self.CAUSAL_MASK_NAME}_mul"

        for node in new_nodes:
            graph.add_node(node)
            self.log_match(node)

        model.graph.initializer.extend(
            [ones_initializer, floating_point_limit_initializer]
        )
        _LOGGER.info(f"Successfully adjusted the {self.CAUSAL_MASK_NAME} input")

        return model
