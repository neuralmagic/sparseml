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

from copy import deepcopy
from typing import Any, Dict, List

from onnx import ModelProto, NodeProto, TensorProto, ValueInfoProto, helper

from sparseml.exporters.transforms.onnx_transform import OnnxTransform
from sparseml.exporters.transforms.utils.matching import get_structural_matches
from sparseml.onnx.utils.graph_editor import ONNXGraph


__all__ = ["AdditionalTransformsBase"]


class AdditionalTransformsBase(OnnxTransform):

    POSITIONS_NAME = "positions"
    CAUSAL_MASK_NAME = "causal_mask"

    def add_causal_mask_input(self, model: ModelProto) -> ModelProto:
        """
        Adds causal mask as an input to the model

        :param model: model to update
        :return: updated model
        """
        input_ids = self._get_input_proto(model, "input_ids")
        attention_mask = self._get_input_proto(model, "attention_mask")

        batch_size = input_ids.type.tensor_type.shape.dim[0].dim_param
        input_ids_length = input_ids.type.tensor_type.shape.dim[1].dim_value
        sequence_length = attention_mask.type.tensor_type.shape.dim[1].dim_value
        causal_mask_input = helper.make_tensor_value_info(
            name=self.CAUSAL_MASK_NAME,
            elem_type=TensorProto.BOOL,
            shape=[batch_size, 1, input_ids_length, sequence_length],
        )
        model.graph.input.append(causal_mask_input)
        return model

    def add_positions_input(self, model: ModelProto) -> ModelProto:
        """
        Adds positions as an input to the model

        :param model: model to update
        :return: updated model
        """
        # positions tensor should have shape equal to input_ids
        input_ids = self._get_input_proto(model, "input_ids")
        positions_input = deepcopy(input_ids)
        positions_input.name = self.POSITIONS_NAME
        model.graph.input.append(positions_input)
        return model

    def find_nodes_by_pattern(
        self, model: ModelProto, pattern: Dict[str, Any]
    ) -> List[NodeProto]:
        """
        Find the node that creates the `position_ids` tensor
        :param model: the ONNX model
        :return: the node that creates the `position_ids` tensor
        """
        graph = ONNXGraph(model)
        matches = get_structural_matches(graph, **pattern)
        if not matches:
            raise ValueError(f"Unable to find pattern:\n{pattern}\nin model")
        return [match.node for match in matches]

    def _get_input_proto(self, model: ModelProto, input_name: str) -> ValueInfoProto:
        """
        Get the input_ids tensor from the model

        :param model: the ONNX model
        :return: the input_ids tensor
        """
        input_ids = [
            input_info
            for input_info in model.graph.input
            if input_info.name == input_name
        ][0]
        if not input_ids:
            raise RuntimeError(
                f"{self.__name__} - unable to find 'input_ids' in model input"
            )
        return input_ids
