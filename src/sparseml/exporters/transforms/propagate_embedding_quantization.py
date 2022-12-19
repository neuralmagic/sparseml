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

import numpy
from onnx import ModelProto, numpy_helper

from sparseml.exporters.transforms.onnx_transform import OnnxTransform
from sparseml.exporters.transforms.utils.matching import get_structural_matches
from sparseml.onnx.utils.graph_editor import ONNXGraph


__all__ = ["PropagateEmbeddingQuantization"]

_LOGGER = logging.getLogger(__name__)


class PropagateEmbeddingQuantization(OnnxTransform):
    """
    A transform for propagating embedding quantizations through concat

    Transforms
    ```
    | uint8 initializer   input
    |           |        |
    |           Gather
    |             |
    |       DequantizeLinear
    |        |   |   |
    |   Slice Slice  |
    |        |   |   |
    |       Pad Pad  |
    |        |   |   |
    |          Concat
    |            |
    |          output
    ```
    Into
    ```
    | uint8 initializer   input
    |           |        |
    |           Gather
    |        |   |   |
    |   Slice Slice  |
    |        |   |   |
    |       Pad Pad  |
    |        |   |   |
    |          Concat
    |             |
    |       DequantizeLinear
    |             |
    |           output
    ```
    """

    def transform(self, model: ModelProto) -> ModelProto:
        graph = ONNXGraph(model)
        matches = get_structural_matches(
            graph,
            op_type="DequantizeLinear",
            parent_ops=[["Gather"]],
            children_ops=[
                ["Slice", "Pad", "Concat"],
                ["Slice", "Pad", "Concat"],
                # NOTE: since the other two branch also send output to concat,
                #       the actual concat branch will be last in a sorted graph.
                ["Concat"],
            ],
        )
        for match in matches:
            (gather,) = match.parents[0]
            dequant = match.node
            slice1, _, concat1 = match.children[0]
            slice2, _, concat2 = match.children[1]
            (concat,) = match.children[2]

            # check for uint8 initializer
            indices = graph.get_init_by_name(gather.input[0])
            if indices is None or numpy_helper.to_array(indices).dtype != numpy.uint8:
                continue

            # check that all concats are the same
            if concat.name != concat1.name or concat.name != concat2.name:
                continue

            self.log_match(match)

            assert concat.input[2] == dequant.output[0]
            concat.input[2] = gather.output[0]
            slice1.input[0] = gather.output[0]
            slice2.input[0] = gather.output[0]

            tmp = concat.output[0]
            concat.output[0] = dequant.output[0]
            dequant.output[0] = tmp
            dequant.input[0] = concat.output[0]
        return model
