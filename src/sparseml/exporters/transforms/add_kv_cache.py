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

from typing import Any, Dict, List

import onnx.helper
from onnx import ModelProto, NodeProto

from sparseml.exporters.transforms import OnnxTransform
from sparseml.exporters.transforms.utils import get_structural_matches
from sparseml.onnx.utils import ONNXGraph


__all__ = ["AddKeyValueCache"]

"""
In CodeGen architecture, the MatMul that performs the V x Softmax(QK^T) operation is 
preceded by two nodes: Cast and Softmax, Thus, we want to find a MatMul node that has two
parent branches: Softmax and Cast, as well as a Transpose node that generates values.
"""
CODEGEN_MATCHING_RULE_VALUE = dict(
    op_type="MatMul", parent_ops=[["Softmax", "Cast"], ["Transpose"]]
)
CODEGEN_VALUE_CACHE_DIMS = ["batch", "num_heads", "past_sequence_len", "hidden_dims"]
CODEGEN_VALUE_CONCAT_AXIS = CODEGEN_VALUE_CACHE_DIMS.index("past_sequence_len")
"""
In CodeGen architecture, the MatMul that performs the Q x K^T operation is 
followed by four nodes: Matmul, Div, Where and Softmax. Thus, K values are coming from the
Transpose node preceding the MatMul
"""
CODEGEN_MATCHING_RULE_KEY = dict(
    op_type="MatMul",
    parent_ops=[["Cast"], ["Transpose"]],
    children_ops=[["Div", "Where", "Add", "Softmax"]],
)
CODEGEN_KEY_CACHE_DIMS = ["batch", "num_heads", "hidden_dims", "past_sequence_len"]
CODEGEN_KEY_CONCAT_AXIS = CODEGEN_KEY_CACHE_DIMS.index("past_sequence_len")


class AddKeyValueCache(OnnxTransform):
    """
    Takes a text generation models (decoder-only types) and adds
    a key and value cache to the model.

    This transformation will add a set of inputs:
        - present.key.0
        - present.value.0
        - present.key.1
        - present.value.1
        ...
    and a set of outputs:
        - past_key_values.key.0
        - past_key_values.value.0
        - past_key_values.key.1
        - past_key_values.value.1
        ...

    So far, this has been only tested on the CodeGen architecture exported
    using SparseML transformer export pipeline.
    """

    def transform(self, model: ModelProto) -> ModelProto:
        graph = ONNXGraph(model)

        # hack, keeping this for now until I figure out how to remove those on export
        [
            model.graph.output.remove(out)
            for out in model.graph.output
            if out.name != "logits"
        ]

        self.add_value_cache(model, graph)
        self.add_key_cache(model, graph)

        return model

    def add_value_cache(
        self,
        model: ModelProto,
        graph: ONNXGraph,
        matching_rule: Dict[str, Any] = CODEGEN_MATCHING_RULE_VALUE,
        concat_axis: List[str] = CODEGEN_VALUE_CONCAT_AXIS,
        cache_dims: int = CODEGEN_VALUE_CACHE_DIMS,
    ):
        """
        Adds a value cache to the model. This means that a Concat node is added
        to the model that concatenates the value of the token that is currently
        being processed with the value cache.

        :param model: The model to add the value cache to
        :param graph: The graph of the model
        :param matching_rule: The rule to use to find the MatMul node that performs
            the V x Softmax(QK^T) operation
        :param concat_axis: The axis to concatenate the cache with values on
        :param cache_dims: The dimensions of the cache
        """

        value_matches = get_structural_matches(graph, **matching_rule)
        if not value_matches:
            raise ValueError("Could not find matching nodes for the key cache. ")

        for match_index, match in enumerate(value_matches):
            self.log_match(match)
            value_node = match.parents[1][0]
            matmul_node = match.node
            self.concatenate_cache_with_model_outputs(
                matmul_node=matmul_node,
                target_node=value_node,
                index=match_index,
                model=model,
                graph=graph,
                concat_axis=concat_axis,
                cache_dims=cache_dims,
                input_to_concat_name=f"past_key_values.{match_index}.value",
                output_from_concat_name=f"present.{match_index}.value",
            )

    def add_key_cache(
        self,
        model: ModelProto,
        graph: ONNXGraph,
        matching_rule: Dict[str, Any] = CODEGEN_MATCHING_RULE_KEY,
        concat_axis: List[str] = CODEGEN_KEY_CONCAT_AXIS,
        cache_dims: int = CODEGEN_KEY_CACHE_DIMS,
    ):
        """
        Adds a key cache to the model. This means that a Concat node is added
        to the model that concatenates the key of the token that is currently
        being processed with the key cache.

        :param model: The model to add the key cache to
        :param graph: The graph of the model
        :param matching_rule: The rule to use to find the MatMul node that performs
            the Q x K^T operation
        :param concat_axis: The axis to concatenate the cache with keys on
        :param cache_dims: The dimensions of the cache
        """

        key_matches = get_structural_matches(graph, **matching_rule)
        if not key_matches:
            raise ValueError("Could not find matching nodes for the key cache. ")

        for match_index, match in enumerate(key_matches):
            self.log_match(match)
            key_node = match.parents[1][0]
            matmul_node = match.node

            self.concatenate_cache_with_model_outputs(
                matmul_node=matmul_node,
                target_node=key_node,
                index=match_index,
                model=model,
                graph=graph,
                concat_axis=concat_axis,
                cache_dims=cache_dims,
                input_to_concat_name=f"past_key_values.{match_index}.key",
                output_from_concat_name=f"present.{match_index}.key",
            )

    def concatenate_cache_with_model_outputs(
        self,
        matmul_node: NodeProto,
        target_node: NodeProto,
        index: int,
        model: ModelProto,
        graph: ONNXGraph,
        concat_axis: int,
        cache_dims: List[str],
        input_to_concat_name: str,
        output_from_concat_name: str,
    ):
        """
        Insert a Concat node into the model that:
         - adds the cache input as an input to the model
         - concatenates the output of the target node with the cache input
         - adds the concatenation as an output to the model
         - replaces the input of the MatMul node with the output of the Concat node

        """

        input_to_matmul = target_node.output[0]
        input_index_matmul = [
            i for i, x in enumerate(matmul_node.input) if x == input_to_matmul
        ][0]
        matmul_node.input[input_index_matmul] = output_from_concat_name

        input_to_concat = onnx.helper.make_tensor_value_info(
            input_to_concat_name,
            onnx.TensorProto.FLOAT,
            cache_dims,
        )
        output_from_concat = onnx.helper.make_tensor_value_info(
            output_from_concat_name,
            onnx.TensorProto.FLOAT,
            cache_dims,
        )

        is_key = input_to_concat_name.endswith("key")

        concat_node = onnx.helper.make_node(
            op_type="Concat",
            inputs=[input_to_concat_name, input_to_matmul],
            outputs=[output_from_concat_name],
            axis=concat_axis,
            name=f"Concat_Key_{index}" if is_key else f"Concat_Value_{index}",
        )

        model.graph.node.insert(
            [
                i
                for i, n in enumerate(graph._model.graph.node)
                if n.name == matmul_node.name
            ][0],
            concat_node,
        )
        model.graph.input.insert(-1, input_to_concat)
        model.graph.output.insert(-1, output_from_concat)
        self.add_node_deferred(concat_node)
