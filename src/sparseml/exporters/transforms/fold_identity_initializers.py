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

from onnx import ModelProto

from sparseml.exporters.transforms import OnnxTransform
from sparseml.exporters.transforms.utils import get_structural_matches
from sparseml.onnx.utils import ONNXGraph


__all__ = ["FoldIdentityInitializers"]

_LOGGER = logging.getLogger(__name__)


def fold_identity_initializer(
    match: "MatchResult", model: ModelProto
) -> ModelProto:  # noqa F821
    """
    Find any node in the graph that uses the output of
    the `match.node` as it's input. Replace the input
    of this node with `match.node`'s input. Finally,
    remove `match.node` from the graph.

    :param match: Match node to be folded into the graph
    :param model: ONNX model to be transformed
    :return: ONNX model with the match node folded into the graph
    """
    graph = ONNXGraph(model)
    for child_node in graph.get_node_children(match.node):
        for i, child_node_input in enumerate(child_node.input):
            if child_node_input == match.node.output[0]:
                child_node.input[i] = match.node.input[0]
        model.graph.node.remove(match.node)
    return model


class FoldIdentityInitializers(OnnxTransform):
    """
    Folds any `Identity` initializer node into the graph.
    TODO: Add graph
    """

    def transform(self, model: ModelProto) -> ModelProto:
        count_converted_nodes = 0
        graph = ONNXGraph(model)
        for match in get_structural_matches(
            graph,
            op_type="Identity",
        ):
            _LOGGER.debug(f"Matched Identity node: {match.node.name}")
            model = fold_identity_initializer(match, model)
            count_converted_nodes += 1

        if count_converted_nodes > 0:
            _LOGGER.info(f"Folded {count_converted_nodes} identity initializer nodes")
        return model
