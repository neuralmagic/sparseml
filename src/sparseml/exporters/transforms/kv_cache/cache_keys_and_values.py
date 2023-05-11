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
from typing import List, Optional, Set, Tuple

from onnx import ModelProto, NodeProto

from sparseml.exporters.transforms.onnx_transform import OnnxTransform
from sparseml.onnx.utils import ONNXGraph


__all__ = ["CacheKeysAndValues"]


_LOGGER = logging.getLogger(__name__)


# idx of key node to the attention scores matmul
# no great way to generically infer this from the graph since transposes can
# be used to place it on either side of the matmul
# hardcoding for now, will update to have a hardcoded value for each model type
_KEY_NODE_INPUT_IDX = 0


class CacheKeysAndValues(OnnxTransform):
    """
    Performs the following updates to a graph to add a basic KV cache mechanism

    1. Adds all matched transformer keys and values as outputs to the graph so they
        may be cached by the model user
    2. Adds those keys and values back as inputs to the graph for update
    3. Adds concat nodes to the graph to apply any cached keys and values before
        they are used in computation

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

        # first matmul: RHS key
        # second matmul: LHS value
        graph = ONNXGraph(model)

        # MATCH -  the matmuls that use the keys and values
        # (attention scores and context layer)
        attention_scores_context_layer_matmul_pairs = (
            _find_attention_scores_context_layer_matmul_pairs(graph)
        )

        # INJECT - cache inputs/outputs + concatenation to the matmuls
        for idx, (attention_scores_matmul, context_layer_matmul) in enumerate(
            attention_scores_context_layer_matmul_pairs
        ):
            cache_layer_name = f"cache.{idx}"

            key_input_idx = _KEY_NODE_INPUT_IDX
            value_input_idx = _value_input_idx(context_layer_matmul, graph)

            # TODO: inject concat node, input, and output for the matmuls for
            # their respective keys and values

        return _find_attention_scores_context_layer_matmul_pairs(graph)


def _find_attention_scores_context_layer_matmul_pairs(
    graph: ONNXGraph,
) -> List[Tuple[NodeProto, NodeProto]]:
    attention_scores_context_layer_matmul_pairs = []

    context_layer_matmuls = [
        node for node in graph.nodes if _is_context_layer_matmul(node, graph)
    ]
    context_layer_matmul_names = {node.name for node in context_layer_matmuls}

    for context_layer_matmul in context_layer_matmuls:
        attention_scores_matmul = _find_attention_scores_matmul_from_context_matmul(
            context_layer_matmul, graph, context_layer_matmul_names
        )
        if attention_scores_matmul is not None:
            attention_scores_context_layer_matmul_pairs.append(
                (attention_scores_matmul, context_layer_matmul)
            )
    return attention_scores_context_layer_matmul_pairs


def _is_context_layer_matmul(node: NodeProto, graph: ONNXGraph) -> bool:
    # candidate context layer matmuls are non-parameterized matmuls where ont of
    # the matmul inputs is the output of a Softmax
    if not is_matmul(node) or _is_parameterized_node(node, graph):
        # not a matmul or MatMul op has a parameter
        return False

    for parent in graph.get_node_parents(node):
        if not isinstance(parent, NodeProto):
            continue
        if parent.op_type == "Softmax":
            # a parent is a Softmax node, assume this is a context layer
            return True

    # no parents are a softmax node
    return False


def _find_attention_scores_matmul_from_context_matmul(
    context_layer_matmul: NodeProto,
    graph: ONNXGraph,
    context_layer_matmul_names: Set[str],
) -> Optional[NodeProto]:
    # perform a BFS up the model DAG from the context layer matmul until
    # we find the corresponding attention score matmul
    # the attention score matmul is assumed to be the first non-paramterized
    # matmul we reach
    # we return None if no such matmul is found, or there is an indication that
    # we have traversed outside the self attention module (found another
    # context layer matmul)

    seen_node_names = {context_layer_matmul.name}
    node_queue = [context_layer_matmul]

    while node_queue:
        current_node = node_queue.pop(0)

        node_parents = graph.get_node_parents(current_node)

        if (
            is_matmul(current_node)
            and (current_node.name != context_layer_matmul.name)
            and not _is_parameterized_node(current_node, graph)
        ):
            # treat root node as regular, non matmul node
            if current_node.name in context_layer_matmul_names:
                _LOGGER.info(
                    "first MatMul node found for context matmul %s "
                    "was another context matmul %s",
                    context_layer_matmul.name,
                    current_node.name,
                )
                return None
            else:
                # success case - first found matmul is non-parameterized
                return current_node

        for parent in node_parents:
            if not isinstance(parent, NodeProto):
                continue
            if parent.name not in seen_node_names:
                seen_node_names.add(parent.name)
                node_queue.append(parent)

    # no matmul matched before bottoming
    _LOGGER.info(
        "No attention scores matmul found for context layer matmul %s",
        context_layer_matmul.name,
    )
    return None


def _value_input_idx(context_layer_matmul: NodeProto, graph: ONNXGraph):
    # get idx of matmul that the value node is an input of
    softmax_input_idx = 0  # default to softmax being on left hand side
    for idx, parent in enumerate(graph.get_node_parents(context_layer_matmul)):
        if isinstance(parent, NodeProto) and parent.op_type == "Softmax":
            softmax_input_idx = idx
            break
    return 1 - softmax_input_idx  # return index that isn't the softmax


def is_matmul(node: NodeProto):
    # matches against FP32 or INT8 matmul types
    return node.op_type in ["MatMul", "MatMulInteger", "Gemm", "QLinearMatMul"]


def _is_parameterized_node(node: NodeProto, graph: ONNXGraph) -> bool:
    # returns true if any input to the node is a parameter (initializer) of the graph
    return any(graph.get_init_by_name(node_input) for node_input in node.input)


"""
import onnx
from sparseml.exporters.transforms.kv_cache import *
model = onnx.load("/home/benjamin/tmp-models/small_decoder_opt.onnx", load_external_data=False)
pairs = CacheKeysAndValues().transform(model)
len(pairs)
"""
