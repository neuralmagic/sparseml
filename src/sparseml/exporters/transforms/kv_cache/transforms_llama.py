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
import warnings

import numpy
import onnx
from onnx import ModelProto, numpy_helper

from sparseml.exporters.transforms.kv_cache.transforms_base import (
    AdditionalTransformsBase,
)
from sparseml.onnx.utils.graph_editor import ONNXGraph
from sparseml.onnx.utils.helpers import get_init_by_name, get_nodes_by_input_id


__all__ = ["AdditionalTransformsLLAMA"]

_LOGGER = logging.getLogger(__name__)


class AdditionalTransformsLLAMA(AdditionalTransformsBase):

    POSITION_IDS_MATCHING_PATTERN = dict(op_type="Range", children_ops=[["Unsqueeze"]])
    CAUSAL_MASK_MATCHING_PATTERN = dict(op_type="Expand", children_ops=[["Add"]])
    SLICE_MAX_INT_NAME = "slice_max_int"

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
        The Slice nodes are identified based on the `data` operator which does not have
        a parent input (as identified using the `get_node_single_parent` function).

        :param model: model to update
        :return: updated model with Slice nodes in the attention heads updated
        """
        arr = numpy.array(numpy.iinfo(numpy.intp).max).reshape(
            1,
        )
        max_int_tensor = numpy_helper.from_array(arr, name=self.SLICE_MAX_INT_NAME)

        nodes_found = 0
        for node in model.graph.node:
            valid_node = False

            if node.op_type == "Slice":
                init = get_init_by_name(model, node.input[0])
                data_parent = ONNXGraph(model).get_node_single_parent(node, 0)
                # The Slice nodes may have data which are initializers or constants
                if init is not None or (
                    data_parent is not None and len(data_parent.input) == 0
                ):
                    valid_node = True

            if valid_node:
                nodes_found += 1
                node.input[2] = self.SLICE_MAX_INT_NAME
                self.log_match(node)

        valid_node_counts = [64, 80]
        if nodes_found not in valid_node_counts:
            warnings.warn(
                f"Number of Slice nodes updated {nodes_found} does not match the "
                f"expected values {valid_node_counts} for the 7 billion or 13 billion "
                "parameter models."
            )

        _LOGGER.info(f"Found {nodes_found} Slice nodes to update")

        model.graph.initializer.append(max_int_tensor)
        ONNXGraph(model).delete_orphaned_node_branches()
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
