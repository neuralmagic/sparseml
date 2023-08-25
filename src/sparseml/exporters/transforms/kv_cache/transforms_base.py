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

from onnx import ModelProto, NodeProto, TensorProto, ValueInfoProto, helper

from sparseml.exporters.transforms.onnx_transform import OnnxTransform
from sparseml.exporters.transforms.utils.matching import get_structural_matches
from sparseml.onnx.utils.graph_editor import ONNXGraph


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
