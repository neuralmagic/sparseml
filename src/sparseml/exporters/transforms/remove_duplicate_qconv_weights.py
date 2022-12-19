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
from typing import List

import numpy
from onnx import ModelProto, numpy_helper

from sparseml.exporters.transforms.onnx_transform import OnnxTransform
from sparseml.onnx.utils import ONNXGraph


__all__ = ["RemoveDuplicateQConvWeights"]

_LOGGER = logging.getLogger(__name__)


class RemoveDuplicateQConvWeights(OnnxTransform):
    """
    Deduplicates weight initializers of QLinearConv and ConvInteger
    that are exactly the same.
    """

    def transform(self, model: ModelProto) -> ModelProto:
        graph = ONNXGraph(model)

        # collect all weights from QLinearConv/ConvInteger nodes
        inits_and_nodes = []
        for node in model.graph.node:
            weight_init = None
            if node.op_type == "QLinearConv":
                weight_init = graph.get_init_by_name(node.input[3])
            elif node.op_type == "ConvInteger":
                weight_init = graph.get_init_by_name(node.input[1])

            if weight_init is not None:
                inits_and_nodes.append((weight_init, node))

        # group them based on weight equality
        weights_for_group: List[numpy.ndarray] = []
        qconv_groups: List[List[str]] = []
        for init, node in inits_and_nodes:
            weight = numpy_helper.to_array(init)
            found_match = False
            for idx, val in enumerate(weights_for_group):
                if val.shape == weight.shape and numpy.all(val == weight):
                    qconv_groups[idx].append(node)
                    found_match = True
                    break
            if not found_match:
                qconv_groups.append([node])
                weights_for_group.append(weight)

        # create the new shared initializers and update the nodes
        num_inits = 0
        num_groups = 0
        for idx, (weight, qconv_nodes) in enumerate(
            zip(weights_for_group, qconv_groups)
        ):
            if len(qconv_nodes) == 1:
                continue

            shared_init = numpy_helper.from_array(
                weight, name=f"qconv_shared_weight_group_{idx}"
            )
            model.graph.initializer.append(shared_init)
            for node in qconv_nodes:
                if node.op_type == "QLinearConv":
                    node.input[3] = shared_init.name
                elif node.op_type == "ConvInteger":
                    node.input[1] = shared_init.name
            node_names = [n.name for n in qconv_nodes]
            _LOGGER.debug("Combined weight initializer for %s", node_names)
            num_inits += len(qconv_nodes)
            num_groups += 1
        _LOGGER.debug("Merged %d weight initializers into %d", num_inits, num_groups)
        return model
