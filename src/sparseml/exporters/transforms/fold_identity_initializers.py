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

from typing import Union

from onnx import ModelProto, NodeProto

from sparseml.exporters.transforms import BaseTransform
from sparseml.onnx.utils import ONNXGraph, check_load_model, validate_onnx_file


def is_identity(node: NodeProto, graph: "ONNXGraph") -> bool:
    return (
        node.op_type == "Identity"
        and len(node.input) == 1
        and len(node.output) == 1
        and node.input[0] in graph._name_to_initializer
    )


class FoldIdentityInitializers(BaseTransform):
    """
    Folds any `Identity` initializer node. Such a node is defined by:
     - having a single input
     - having a single output
     - being an `Identity` operation
     - being an `initializer` node
    """

    def _transform(self, model: ModelProto) -> ModelProto:
        graph = ONNXGraph(model)
        matches = []

        for node in model.graph.node:
            if not is_identity(node, graph):
                continue
            matches.append(node)

            # find any node in the graph that uses the output of `node`
            # as an input.
            # replace the input with `node`'s input
            for other in graph.get_node_children(node):
                for i, other_input_i in enumerate(other.input):
                    # NOTE: this just replaces the str ids
                    if other_input_i == node.output[0]:
                        other.input[i] = node.input[0]

        for node in matches:
            model.graph.node.remove(node)
        return model

    def _validate_input(self, model: ModelProto):
        validate_onnx_file(model)

    def _validate_output(self, model: ModelProto):
        validate_onnx_file(model)

    def __call__(self, model: Union[ModelProto, str]) -> ModelProto:
        onnx_model = check_load_model(model)
        self._validate_input(onnx_model)
        onnx_model = self._transform(onnx_model)
        self._validate_output(onnx_model)
        return onnx_model
