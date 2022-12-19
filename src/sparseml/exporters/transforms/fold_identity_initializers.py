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
from onnx import ModelProto

from sparseml.exporters.transforms import OnnxTransform
from sparseml.exporters.transforms.utils import get_structural_matches
from sparseml.onnx.utils import ONNXGraph


__all__ = ["FoldIdentityInitializers"]


class FoldIdentityInitializers(OnnxTransform):
    """
    Removes `Identity` nodes.

    Transforms
    ```
    | initializer
    |     |
    | Identity
    |     |
    | (Anything)
    ```

    into
    ```
    | initializer
    |     |
    | (Anything)
    ```
    """

    def transform(self, model: ModelProto) -> ModelProto:
        graph = ONNXGraph(model)
        for match in get_structural_matches(graph, op_type="Identity"):
            self.log_match(match)
            for child_node in graph.get_node_children(match.node):
                for i, child_node_input in enumerate(child_node.input):
                    if child_node_input == match.node.output[0]:
                        child_node.input[i] = match.node.input[0]
            self.delete_node_deferred(match.node)
        return model
