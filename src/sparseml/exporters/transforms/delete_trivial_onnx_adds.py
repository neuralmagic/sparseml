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
from sparseml.exporters.transforms.utils import assert_node_type, get_structural_matches
from sparseml.onnx.utils import ONNXGraph, remove_node_and_params_from_graph


_LOGGER = logging.getLogger(__name__)


def should_delete_trivial_onnx_add(match: "MatchResult") -> bool: # F821
    """
    Check if the match node's second input
    node is a constant node set to zero.

    :param match: Match node to be folded into the graph
    :return: Boolean flag indicating whether the
        add node should be deleted
    """
    constant_node = match.parents[1][0]
    if not assert_node_type(constant_node, "Constant"):
        return False
    constant_node_add_value = numpy_helper.to_array(constant_node.attribute[0].t)
    if not numpy.all(constant_node_add_value == 0.0):
        return False
    return True


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
        graph = ONNXGraph(model)
        for match in get_structural_matches(
            graph, op_type="Add", parent_ops=[[], ["Constant"]]
        ):

            if should_delete_trivial_onnx_add(match):
                _LOGGER.debug(f"Matched Identity node: {match.node.name}")
                constant_node = match.parents[1][0]
                remove_node_and_params_from_graph(model, match.node)
                remove_node_and_params_from_graph(model, constant_node)
                count_converted_nodes += 1

        if count_converted_nodes > 0:
            _LOGGER.info(f"Delete {count_converted_nodes} trivial ONNX Add nodes")
        return model
