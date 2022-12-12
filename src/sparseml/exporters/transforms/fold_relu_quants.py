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

from onnx import ModelProto, numpy_helper

from sparseml.exporters.transforms import OnnxTransform
from sparseml.exporters.transforms.utils import get_structural_matches
from sparseml.onnx.utils import (
    ONNXGraph,
    get_init_by_name,
    remove_node_and_params_from_graph,
)


__all__ = ["FoldReLUQuants"]

_LOGGER = logging.getLogger(__name__)


class FoldReLUQuants(OnnxTransform):
    """
    Transforms

    ```
    |   INPUT
    |     |
    |   ReLU   scale   zero point (missing or == 0)
    |      |     |      |
    |       QuantizeLinear (with zero point of 0)
    |            |
    |          OUTPUT
    ```

    into

    ```
    |   INPUT   scale   zero point (missing or == 0)
    |      |     |      |
    |       QuantizeLinear (with zero point of 0)
    |            |
    |          OUTPUT
    ```
    """

    def transform(self, model: ModelProto) -> ModelProto:
        matches = get_structural_matches(
            ONNXGraph(model), op_type="QuantizeLinear", parent_ops=[["Relu"]]
        )
        for match in matches:
            quant = match.node
            (relu,) = match.parents[0]

            if len(quant.input) == 3:
                zero_point = get_init_by_name(model, quant.input[2])
                zero_point = numpy_helper.to_array(zero_point)
                if zero_point != 0:
                    continue

            _LOGGER.debug(f"Matched {match}")
            quant.input[0] = relu.input[0]
            remove_node_and_params_from_graph(model, relu)
        return model
