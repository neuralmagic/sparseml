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

from onnx import ModelProto, NodeProto

from sparseml.exporters.transforms.onnx_transform import OnnxTransform
from sparseml.onnx.utils import ONNXGraph


_LOGGER = logging.getLogger(__name__)


class SkipInputQuantize(OnnxTransform):
    def transform(self, model: ModelProto) -> ModelProto:
        if (
            len(model.graph.input) != 1
            or model.graph.input[0].type.tensor_type.elem_type != 1
        ):
            # more than 1 input or input is not FP32
            return (
                "Not modifying ONNX graph inputs - either graph has more than one "
                "input or input type is not FP32"
            )

        input_node = model.graph.input[0]
        input_children = [
            node for node in model.graph.node if input_node.name in node.input
        ]
        if not all(node.op_type == "QuantizeLinear" for node in input_children):
            return (
                "Not modifying ONNX graph inputs - only QuantizeLinear nodes may follow"
                "the FP32 input tensor in original graph, prior to converting to uint8"
            )

        _delete_quantize_nodes(ONNXGraph(model), input_children)
        input_node.type.tensor_type.elem_type = 2  # fp32 -> uint8
        _LOGGER.info(
            "Model initial QuantizeLinear node(s) deleted and inputs set to uint8"
        )

        return None


def _delete_quantize_nodes(graph: ONNXGraph, quantize_nodes: List[NodeProto]):
    # delete given quantize nodes and forward their inputs to the next graph layer
    for quantize_node in quantize_nodes:
        quantize_children = graph.get_node_children(quantize_node)
        quantize_node_id = quantize_node.output[0]
        for child_node in quantize_children:
            input_idx = [
                idx
                for idx, inp in enumerate(child_node.input)
                if inp == quantize_node_id
            ]
            if not input_idx:
                continue
            input_idx = input_idx[0]
            graph.update_node_input(child_node, quantize_node.input[0], input_idx)
            _LOGGER.debug(
                f"set node with output id {child_node.output[0]} as initial node in "
                "graph"
            )

    _LOGGER.debug(
        f"deleting QuantizeLinear node(s) with output id(s): "
        f"{[n.output for n in quantize_nodes]}"
    )
    graph.delete_nodes(quantize_nodes)  # only contains references to the Quantize nodes
    graph.delete_unused_initializers()  # cleanup
