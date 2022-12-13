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

from sparseml.exporters.transforms import OnnxTransform
from sparseml.exporters.transforms.utils import get_structural_matches
from sparseml.onnx.utils import ONNXGraph, remove_node_and_params_from_graph


_LOGGER = logging.getLogger(__name__)


class DeleteTrivialOnnxAdds(OnnxTransform):
    """
    | Starting with:
    |   Input   Constant (with initializer, optionally set to zero)
    |       |    |
    |        ADD
    |         |
    |       OUTPUT
    |
    | We end up converting to:
    |        Input
    |         |
    |       OUTPUT
    """

    def transform(self, model: ModelProto) -> ModelProto:
        count_converted_nodes = 0
        for match in get_structural_matches(
            ONNXGraph(model), op_type="Add", parent_ops=[[], ["Constant"]]
        ):
            add = match.node
            (constant,) = match.parents[1]

            constant_array = numpy_helper.to_array(constant.attribute[0].t)
            if not numpy.all(constant_array == 0.0):
                continue

            _LOGGER.debug(f"Matched Identity node: {match.node.name}")
            count_converted_nodes += 1

            graph = ONNXGraph(model)
            parent = graph.get_node_single_parent(add, 0)
            if parent is not None:
                parent.output[0] = add.output[0]
            remove_node_and_params_from_graph(model, add)
            remove_node_and_params_from_graph(model, constant)

        if count_converted_nodes > 0:
            _LOGGER.info(f"Delete {count_converted_nodes} trivial ONNX Add nodes")
        return model
