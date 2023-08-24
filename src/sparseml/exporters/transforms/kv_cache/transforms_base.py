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
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy
import onnx
from onnx import ModelProto, NodeProto, TensorProto, ValueInfoProto, helper

from sparseml.exporters.transforms.onnx_transform import OnnxTransform
from sparseml.exporters.transforms.utils.matching import get_structural_matches
from sparseml.onnx.utils.graph_editor import ONNXGraph
from sparseml.onnx.utils.helpers import get_nodes_by_input_id


__all__ = ["AdditionalTransformsBase"]

_LOGGER = logging.getLogger(__name__)


class AdditionalTransformsBase(OnnxTransform):

    POSITIONS_NAME = "positions"
    CAUSAL_MASK_NAME = "causal_mask"

    def add_causal_mask_input(self, model: ModelProto) -> ModelProto:
        """
        Adds causal mask as an input to the model.

        Causal mask is an int64 tensor of shape
        [batch_size, 1, input_ids_length, sequence_length]
        where the value is 0 if the position is masked and 1
        otherwise.

        :param model: model to update
        :return: updated model
        """
        input_ids = self._get_input_proto(model, "input_ids")
        attention_mask = self._get_input_proto(model, "attention_mask")

        batch_size = input_ids.type.tensor_type.shape.dim[0].dim_param
        input_ids_length = input_ids.type.tensor_type.shape.dim[1].dim_value
        sequence_length = attention_mask.type.tensor_type.shape.dim[1].dim_param

        causal_mask_input = helper.make_tensor_value_info(
            name=self.CAUSAL_MASK_NAME,
            elem_type=TensorProto.INT64,
            shape=[batch_size, 1, input_ids_length, sequence_length],
        )
        model.graph.input.append(causal_mask_input)
        _LOGGER.info(f"Inserted {self.CAUSAL_MASK_NAME} input to the ONNX model")
        return model

    def add_positions_input(self, model: ModelProto) -> ModelProto:
        """
        Adds positions as an input to the model.

        Positions is a tensor of shape and dtype
        equal to input_ids.

        :param model: model to update
        :return: updated model
        """
        input_ids = self._get_input_proto(model, "input_ids")
        positions_input = deepcopy(input_ids)
        positions_input.name = self.POSITIONS_NAME
        model.graph.input.append(positions_input)
        _LOGGER.info(f"Inserted {self.POSITIONS_NAME} input to the ONNX model")
        return model

    def find_nodes_by_pattern(
        self, model: ModelProto, pattern: Dict[str, Any]
    ) -> List[NodeProto]:
        """
        Find the all the nodes in the `model` that
        match the specified `pattern`.

        :param model: the ONNX model
        :param pattern: a dictionary of arguments and variables
            expected by the `get_structural_matches` function. For
            more information, see the documentation for that function.
        :return: a list of nodes that match the pattern
        """
        graph = ONNXGraph(model)
        matches = get_structural_matches(graph, **pattern)
        if not matches:
            raise ValueError(f"Unable to find pattern:\n{pattern}\nin model")
        return [match.node for match in matches]

    def inject_causal_mask(
        self,
        model: ModelProto,
        nodes: List[NodeProto],
        nodes_parent_op_type: Optional[str] = None,
    ) -> ModelProto:
        """
        Injects causal mask to the graph, replacing the specified nodes.

        :param model: the ONNX model to inject the causal mask into
        :param nodes: the nodes to replace with the causal mask
        :param nodes_parent_op_type: the parent op type of the nodes to replace

        :return: the updated model
        """

        return self.swap_nodes_for_input(
            model, nodes, self.CAUSAL_MASK_NAME, nodes_parent_op_type
        )

    def inject_positions(
        self,
        model: ModelProto,
        nodes: List[NodeProto],
        nodes_parent_op_type: Optional[str] = None,
    ) -> ModelProto:
        """
        Injects positions to the graph, replacing the specified nodes.

        :param model: the ONNX model to inject the positions into
        :param nodes: the nodes to replace with the positions
        :param nodes_parent_op_type: the parent op type of the nodes to replace

        :return: the updated model
        """

        return self.swap_nodes_for_input(
            model, nodes, self.POSITIONS_NAME, nodes_parent_op_type
        )

    def swap_nodes_for_input(
        self,
        model: ModelProto,
        nodes: List[NodeProto],
        input_name: str,
        nodes_parent_op_type: Optional[str] = None,
    ) -> ModelProto:

        """
        Injects the specified input to the graph, replacing the specified nodes.

        :param model: the ONNX model to inject the input into
        :param nodes: the nodes to replace with the input
        :param input_name: the name of the input to replace the nodes with
        :param nodes_parent_op_type: the parent op type of the nodes to replace

        :return: the updated model
        """

        graph = ONNXGraph(model)
        for node in nodes:
            child_node = graph.get_node_children(node)[0]

            if nodes_parent_op_type:
                assert child_node.op_type == nodes_parent_op_type, (
                    f"Expected to find {nodes_parent_op_type} node, "
                    f"found {child_node.op_type}"
                )
            output_to_replace = node.output[0]
            self.log_match(node)
            for idx, input_name_child_node in enumerate(child_node.input):
                if input_name_child_node == output_to_replace:
                    graph.update_node_input(child_node, input_name, idx)

        graph.delete_orphaned_node_branches()

        _LOGGER.info(
            f"Successfully swapped {len(nodes)} nodes for input '{input_name}'"
        )

        return model

    def _get_input_proto(self, model: ModelProto, input_name: str) -> ValueInfoProto:
        """
        Get the input proto for the specified input name.

        :param model: the ONNX model
        :param input_name: the name of the input
        """
        input_proto = [
            input_info
            for input_info in model.graph.input
            if input_info.name == input_name
        ][0]
        if not input_proto:
            raise RuntimeError(
                f"{self.__name__} - unable to find '{input_name}' in model input"
            )
        return input_proto

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
