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

from sparseml.exporters.transforms.onnx_transform import OnnxTransform
from sparseml.exporters.transforms.utils import assert_node_type
from sparseml.onnx.utils import ONNXGraph


__all__ = ["RemoveDuplicateCastOps"]


class RemoveDuplicateCastOps(OnnxTransform):
    """
    Removes duplicate Cast ops

    Transforms:
    ```
    |   input
    |     |
    |   Cast
    |     |
    |   Cast
    ```

    ```
    |   input
    |    |
    |   Cast
    ```
    """

    def transform(self, model: ModelProto) -> ModelProto:
        graph = ONNXGraph(model)
        cast_nodes = [n for n in model.graph.node if n.op_type == "Cast"]
        for cast_node_1 in cast_nodes:
            cast_node_2 = graph.get_node_single_child(cast_node_1)
            if not assert_node_type(cast_node_2, "Cast"):
                continue
            if not cast_node_1.attribute == cast_node_2.attribute:
                continue
            self.log_match(cast_node_1)

            # forward first qat block input to that of the second
            cast_node_2.input[0] = cast_node_1.input[0]

            # remove repeated cast node
            self.delete_node_deferred(cast_node_1)

        return model
