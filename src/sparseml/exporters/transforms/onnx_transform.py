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
from abc import abstractmethod
from typing import Union

from onnx import ModelProto, NodeProto

from sparseml.exporters.transforms import BaseTransform
from sparseml.onnx.utils import ONNXGraph, check_load_model, validate_onnx_file


__all__ = ["OnnxTransform"]

_LOGGER = logging.getLogger(__name__)


class OnnxTransform(BaseTransform):
    """
    Interface that all transforms that operate on ONNX models
    must implement.
    """

    def __init__(self) -> None:
        super().__init__()
        self._nodes_to_delete = []
        self._nodes_to_add = []

    def add_node_deferred(self, node: NodeProto):
        _LOGGER.debug(f"Adding node {node.name} op_type={node.op_type}")
        self._nodes_to_add.append(node)

    def delete_node_deferred(self, node: NodeProto):
        _LOGGER.debug(f"Removing node {node.name} op_type={node.op_type}")
        self._nodes_to_delete.append(node)

    @abstractmethod
    def transform(self, model: ModelProto) -> ModelProto:
        """
        Logic for applying the transformation to the ONNX model

        :param model: The input ONNX model to be transformed
        :return: The transformed ONNX model
        """
        raise NotImplementedError

    def pre_validate(self, model: Union[ModelProto, str]) -> ModelProto:
        """
        Validate the input model before applying the transform.

        :param model: The input ONNX model to be
            validated. It can be a path to the model
            or the model itself.
        :return: The validated ONNX model
        """
        if not ((isinstance(model, str) or isinstance(model, ModelProto))):
            raise ValueError(
                f"Invalid model type: {type(model)}. "
                "Must be a string (path to the .onnx file) or ONNX ModelProto"
            )
        model = check_load_model(model)
        validate_onnx_file(model)
        self._nodes_to_delete.clear()
        self._nodes_to_add.clear()
        return model

    def post_validate(self, model: ModelProto) -> ModelProto:
        """
        Validate the input model after applying the transform
        :param model: The input ONNX model to be validated
        :return The validated ONNX model
        """
        model.graph.node.extend(self._nodes_to_add)
        for node in self._nodes_to_delete:
            model.graph.node.remove(node)
        graph = ONNXGraph(model)
        graph.delete_unused_initializers()
        graph.sort_nodes_topologically()
        validate_onnx_file(model)
        return model
