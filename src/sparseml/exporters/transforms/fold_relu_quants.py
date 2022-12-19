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
from sparseml.exporters.transforms.utils import (
    get_quantization_params,
    get_structural_matches,
)
from sparseml.onnx.utils import ONNXGraph


__all__ = ["FoldReLUQuants"]


class FoldReLUQuants(OnnxTransform):
    """
    Transforms

    ```
    | input
    |   |
    | ReLU   scale   zero point (missing or == 0)
    |    |     |      |
    |     QuantizeLinear
    ```

    into

    ```
    | input   scale   zero point (missing or == 0)
    |    |     |      |
    |     QuantizeLinear
    ```
    """

    def transform(self, model: ModelProto) -> ModelProto:
        graph = ONNXGraph(model)
        for match in get_structural_matches(graph, op_type="Relu"):
            relu_children = graph.get_node_children(match.node)

            delete = True
            for quant in relu_children:
                if quant.op_type != "QuantizeLinear" or (
                    len(quant.input) == 3
                    and get_quantization_params(model, quant).zero_point != 0
                ):
                    delete = False
                    break

            # set all child input nodes to the relu node input
            if delete:
                self.log_match(match)
                for quant in relu_children:
                    quant.input[0] = match.node.input[0]
                self.delete_node_deferred(match.node)
        return model
