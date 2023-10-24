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


class AdditionalTransformsMPT(AdditionalTransformsBase):

    CAUSAL_MASK_MATCHING_PATTERN = dict(op_type="Cast", children_ops=[["Where"]])

    def transform(self, model: ModelProto) -> ModelProto:
        """
        1. Adds `causal_mask` as an input to the model
        2. Finds the nodes that initially create the `causal_mask` tensors
        3. Updates the nodes to use the causal_mask input instead of
              computing it from the Cast op

        :param model: model to update
        :return: updated model
        """
        model = self.add_causal_mask_input(model)
        causal_mask_nodes = self.find_nodes_by_pattern(
            model, pattern=self.CAUSAL_MASK_MATCHING_PATTERN
        )

        model = self.inject_causal_mask(model, causal_mask_nodes, "Where")
        model = self.adjust_causal_mask(model)

        return model

    def adjust_causal_mask(self, model: ModelProto) -> ModelProto:
        """
        Insert a `Cast` and `Not` node after the causal mask input to change
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
        |           Not
        |        (to negate)
        |            |
        |            |
        |   causal_mask_input_child

        The resulting node will change the input int64 mask,
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
            [[[[False, False, False, True, True, True],
                [False, False, False, False, True, True],
                [False, False, False, False, False, True],
                [False, False, False, False, False, False]]]]
        ```

        :param model: the model to update
        :return: the updated model
        """

        graph = ONNXGraph(model)

        cast_node = onnx.helper.make_node(
            "Cast",
            inputs=[self.CAUSAL_MASK_NAME],
            outputs=[f"{self.CAUSAL_MASK_NAME}_cast"],
            to=TensorProto.BOOL,
        )

        not_node = onnx.helper.make_node(
            "Not",
            inputs=[f"{self.CAUSAL_MASK_NAME}_cast"],
            outputs=[f"{self.CAUSAL_MASK_NAME}_not"],
        )

        # get the nodes that take the causal mask as input
        # and replace the input with the adjusted causal mask input
        causal_mask_input_children = get_nodes_by_input_id(model, self.CAUSAL_MASK_NAME)
        for causal_mask_input_child in causal_mask_input_children:
            for idx, input_name in enumerate(causal_mask_input_child.input):
                if input_name == self.CAUSAL_MASK_NAME:
                    causal_mask_input_child.input[idx] = f"{self.CAUSAL_MASK_NAME}_not"

        for node in [cast_node, not_node]:
            graph.add_node(node)
            self.log_match(node)

        _LOGGER.info(f"Successfully adjusted the {self.CAUSAL_MASK_NAME} input")

        return model
