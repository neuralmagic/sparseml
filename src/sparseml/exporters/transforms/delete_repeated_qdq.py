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
from sparseml.exporters.transforms.utils import get_structural_matches
from sparseml.onnx.utils import ONNXGraph


__all__ = ["DeleteRepeatedQdq"]


class DeleteRepeatedQdq(OnnxTransform):
    """
    Removes a QDQ that immediately follows another QDQ.
    NOTE: this will modify graph outputs and does not
    guarantee identical graph behavior. This should be avoided
    and handled with better graph construction.

    Transforms
    ```
    | QuantizeLinear
    |     |
    | DequantizeLinear
    |     |
    | QuantizeLinear
    |     |
    | DequantizeLinear
    ```
    Into
    ```
    | QuantizeLinear
    |     |
    | DequantizeLinear
    ```
    """

    def transform(self, model: ModelProto) -> ModelProto:
        matches = get_structural_matches(
            ONNXGraph(model),
            op_type="QuantizeLinear",
            children_ops=[["DequantizeLinear", "QuantizeLinear", "DequantizeLinear"]],
        )
        for match in matches:
            self.log_match(match)
            quant_node_1 = match.node
            (dequant_node_1, quant_node_2, dequant_node_2) = match.children[0]

            # forward first qat block input to that of the second
            quant_node_2.input[0] = quant_node_1.input[0]

            # remove repeated quant/dequant block
            self.delete_node_deferred(quant_node_1)
            self.delete_node_deferred(dequant_node_1)
        return model
