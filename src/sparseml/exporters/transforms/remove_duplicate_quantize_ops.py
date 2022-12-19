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
from collections import defaultdict

from onnx import ModelProto

from sparseml.exporters.transforms.onnx_transform import OnnxTransform
from sparseml.exporters.transforms.utils import get_quantization_params


__all__ = ["RemoveDuplicateQuantizeOps"]

_LOGGER = logging.getLogger(__name__)


class RemoveDuplicateQuantizeOps(OnnxTransform):
    """
    Removes QuantizeLinear nodes that have the same params.

    Transforms:
    ```
    |   input
    | |   |  ... |
    | Q   Q      Q
    | |   |      |
    ```
    where `Q` are all separate instances of the same QuantizeLinear node, into:
    ```
    |   input
    |    |
    |    Q
    | |  | ... |
    ```
    """

    def transform(self, model: ModelProto) -> ModelProto:
        quantize_ops_by_input = defaultdict(list)
        for node in model.graph.node:
            if node.op_type == "QuantizeLinear":
                quantize_ops_by_input[node.input[0]].append(node)

        # search for nodes to delete
        input_replacements = {}
        for quantize_op_group in quantize_ops_by_input.values():
            keep_node, *remove_nodes = quantize_op_group
            keep_node_params = get_quantization_params(model, keep_node)
            for remove_node in remove_nodes:
                params = get_quantization_params(model, remove_node)
                if keep_node_params == params:
                    input_replacements[remove_node.output[0]] = keep_node.output[0]
                    self.delete_node_deferred(remove_node)

        # replace all the ids of the nodes that are removed
        for old_id, new_id in input_replacements.items():
            for node in model.graph.node:
                for idx, inp in enumerate(node.input):
                    if inp == old_id:
                        node.input[idx] = new_id

        _LOGGER.debug("Removed %d QuantizeLinear nodes", len(self._nodes_to_delete))
        return model
