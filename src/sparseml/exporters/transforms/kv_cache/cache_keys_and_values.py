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

import copy
import logging
from typing import List, Optional, Set, Tuple, Union

import numpy
import onnx
from onnx import ModelProto, NodeProto, TensorProto, ValueInfoProto, numpy_helper

from sparseml.exporters.transforms.onnx_transform import OnnxTransform
from sparseml.onnx.utils import ONNXGraph


__all__ = ["CacheKeysAndValues"]


_LOGGER = logging.getLogger(__name__)

OUTPUT_CACHE_NAME = """present.{attention_layer_idx}.{cache_type}"""
INPUT_CACHE_NAME = """past_key_values.{attention_layer_idx}.{cache_type}"""


class CacheKeysAndValues(OnnxTransform):
    """
    Inject the key and value caches into the graph for the attention layers.
    The logic for pattern matching is as follows:

    1.  Find all the MatMuls that are preceded by a Softmax operation.
        Those are the MatMuls that perform V x Softmax(Q x K^T) operation
        (the "value" MatMuls).
    2.  Given the "value" MatMuls found in step 1, perform a Breadth First Search
        to find the "key" MatMuls that perform Q x K^T operation.
    3.  Before each pair of "key" and "value" MatMuls, inject a cache node that
        concatenates the current keys/values with the cached keys/values.
    4.  For the key cache, the concatenation happens before the Transpose node, that
        precedes the "key" MatMul.
    5.  For the value cache, the concatenation happens directly before the "value"
        MatMul.

    This transform also sets the subset of kv cache inputs/outputs dimensions (
    num_attention_heads and hidden_size_kv_cache ) to the appropriate static values.

    :param num_attention_heads: number of attention heads of the model
    :param hidden_size_kv_cache: hidden size of the key and value cache
    :param internally_multiply_batch_by_num_attention_heads: every model created by
        this transformation, will have kv_cache inputs/outputs that have dimensions:

        [`batch_size`,`num_attention_heads`,`past_sequence_len`,`hidden_size_kv_cache`]

        However, internally, there may be a need of reshaping the kv_cache
        inputs/outputs ("merging" the `batch_size` and `num_attention_heads`
        dimensions) so that it is compatible with the attention layer.
        If True, the batch size will be multiplied
        by the number of attention heads just before the appropriate concatenation node.

    Transforms
    ```
    |
    |     Key
    |      |    Query
    |  Transpose |
    |      |    |
    |       | |
    |        |
    |   "key" MatMul
    |        |
    |       ...    Value
    |        |      |
    |     Softmax  |
    |        |    |
    |       ...  |
    |        |  |
    |         |
    |   "value" MatMul
    |        |
    |       ...
    ```
    to

    ```
    |
    | KeyCache  Key
    |    |      |
    | Reshape   |
    |(optional)|
    |      | |
    |       |
    |   Concat ------------> OutputKeyCache
    |      |
    |      |    Query
    |  Transpose |      ValueCache
    |      |    | Value     |
    |       | |     |    Reshape
    |        |      |  (optional)
    |        |       |  |
    |   "key" MatMul  |
    |        |       |
    |       ...   Concat --> OutputValueCache
    |        |      |
    |     Softmax  |
    |        |    |
    |       ...  |
    |        |  |
    |         |
    |   "value" MatMul
    |        |
    |       ...
    ```

    """

    def __init__(
        self,
        num_attention_heads: int,
        hidden_size_kv_cache: int,
        internally_multiply_batch_by_num_attention_heads: bool = True,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.hidden_size_kv_cache = hidden_size_kv_cache
        self.internally_multiply_batch_by_num_attention_heads = (
            internally_multiply_batch_by_num_attention_heads
        )

    def transform(self, model: ModelProto) -> ModelProto:
        graph = ONNXGraph(model)

        key_value_matmul_pairs = _find_key_value_matmul_pairs(graph)

        # Inject kv cache to the graph as the model input,
        # Inject kv cache concatenated with the current keys/values as the output
        inputs_to_add = []
        outputs_to_add = []

        # get default int8 type to use if graph is quantized
        use_uint8_if_quantized = _use_uint8_if_quantized(graph)

        for idx, (key_matmul, value_matmul) in enumerate(key_value_matmul_pairs):

            value_input_idx = _value_input_idx(value_matmul, model)

            key_concat_node, key_input_tensor, key_output_tensor = create_cache(
                model=model,
                node=key_matmul,
                cache_input_idx=1,
                cache_input_name=INPUT_CACHE_NAME.format(
                    attention_layer_idx=idx, cache_type="key"
                ),
                cache_output_name=OUTPUT_CACHE_NAME.format(
                    attention_layer_idx=idx, cache_type="key"
                ),
                use_uint8_if_quantized=use_uint8_if_quantized,
                num_attention_heads=self.num_attention_heads,
                hidden_size_kv_cache=self.hidden_size_kv_cache,
                multiply_batch_by_num_attention_heads=self.internally_multiply_batch_by_num_attention_heads,  # noqa E501
            )
            value_concat_node, value_input_tensor, value_output_tensor = create_cache(
                model=model,
                node=value_matmul,
                cache_input_idx=value_input_idx,
                cache_input_name=INPUT_CACHE_NAME.format(
                    attention_layer_idx=idx, cache_type="value"
                ),
                cache_output_name=OUTPUT_CACHE_NAME.format(
                    attention_layer_idx=idx, cache_type="value"
                ),
                use_uint8_if_quantized=use_uint8_if_quantized,
                num_attention_heads=self.num_attention_heads,
                hidden_size_kv_cache=self.hidden_size_kv_cache,
                multiply_batch_by_num_attention_heads=self.internally_multiply_batch_by_num_attention_heads,  # noqa E501
            )

            inputs_to_add.extend([key_input_tensor, value_input_tensor])
            outputs_to_add.extend([key_output_tensor, value_output_tensor])

            self.log_match(key_matmul)
            self.log_match(value_matmul)

        # update model with cache inputs, and outputs
        model.graph.input.extend(inputs_to_add)
        model.graph.output.extend(outputs_to_add)

        _set_attention_mask_to_dynamic(model)

        return model


def create_cache(
    model: ModelProto,
    node: NodeProto,
    cache_input_idx: int,
    cache_input_name: str,
    cache_output_name: str,
    num_attention_heads: int,
    hidden_size_kv_cache: int,
    use_uint8_if_quantized: bool = True,
    batch_size: int = 1,
    multiply_batch_by_num_attention_heads: bool = True,
) -> Tuple[NodeProto, ValueInfoProto, ValueInfoProto]:
    """
    Injects a cache (value or key) into the graph for a given Matmul node.

    :param model: Model to update
    :param node: MatMul node that follows the cache injection point
    :param cache_input_idx: Index of the input
        (where the cache will be injected) to the MatMul
    :param cache_input_name: Name of cache input
    :param cache_output_name: Name of cache output
    :param num_attention_heads: number of attention heads of the model
    :param hidden_size_kv_cache: hidden size of the key/value cache
    :param use_uint8_if_quantized: True if quantized MatMuls should have uint8
        inputs, if False, uses int8
    :param batch_size: batch size of the kv cache. By default, this is 1.
    :param multiply_batch_by_num_attention_heads: If True, the batch size of the
        kv cache is multiplied by the number of attention heads before the
        concat node.
    :return: tuple of concat node to add, cache input to add, and cache output to add,
        updates existing nodes in-place
    """
    CACHE_INPUT_DIMS = [
        batch_size,
        num_attention_heads,
        "past_sequence_len",
        hidden_size_kv_cache,
    ]
    CACHE_OUTPUT_DIMS = [
        batch_size,
        num_attention_heads,
        "past_sequence_len + 1",
        hidden_size_kv_cache,
    ]

    graph = ONNXGraph(model)

    cache_data_type = (
        TensorProto.FLOAT
        if node.op_type not in ["MatMulInteger", "QLinearMatMul"]
        else TensorProto.UINT8
        if use_uint8_if_quantized
        else TensorProto.INT8
    )

    # create graph input info proto
    cache_input_info = onnx.helper.make_tensor_value_info(
        cache_input_name,
        cache_data_type,
        CACHE_INPUT_DIMS,
    )
    # create graph output info proto
    cache_output_info = onnx.helper.make_tensor_value_info(
        cache_output_name,
        cache_data_type,
        CACHE_OUTPUT_DIMS,
    )

    if node.op_type == "QLinearMatMul" and cache_input_idx == 1:
        cache_input_idx = 3  # QLinearMatMul B matrix is at idx 3, not 1

    cache_parent = graph.get_node_single_parent(node, index=cache_input_idx)
    if isinstance(cache_parent, NodeProto) and cache_parent.op_type == "Transpose":
        # move cache to before a transpose if applicable
        # this is due to pytorch operations potentially extracting shape values
        # from the key tensor before the transpose is applied
        pre_cache_input_id = cache_parent.input[0]
        # update concat axis
        node = cache_parent
    else:
        pre_cache_input_id = node.input[cache_input_idx]

    cache_input_name_concat = cache_input_name
    cache_output_name_concat = cache_output_name
    cache_input_dims_concat = CACHE_INPUT_DIMS

    if multiply_batch_by_num_attention_heads:
        (
            model,
            cache_input_dims_concat,
            cache_input_name_concat,
            cache_output_name_concat,
        ) = reshape_kv_cache_inputs_outputs(
            model=model,
            cache_input_name=cache_input_name_concat,
            cache_output_name=cache_output_name_concat,
            cache_input_dims=cache_input_dims_concat,
            batch_size=batch_size,
            num_attention_heads=num_attention_heads,
        )

    concat_axis = [
        idx
        for (idx, dim) in enumerate(cache_input_dims_concat)
        if dim == "past_sequence_len"
    ][0]

    concat_node = onnx.helper.make_node(
        op_type="Concat",
        inputs=[cache_input_name_concat, pre_cache_input_id],
        outputs=[cache_output_name_concat],
        axis=concat_axis,
        name=f"concat.{cache_input_name_concat}",
    )

    for node in model.graph.node:
        for input_idx, input_id in enumerate(node.input):
            if input_id == pre_cache_input_id and node.name != concat_node.name:
                node.input[input_idx] = cache_output_name_concat

    graph.add_node(concat_node)

    return concat_node, cache_input_info, cache_output_info


def reshape_kv_cache_inputs_outputs(
    model: ModelProto,
    cache_input_name: str,
    cache_output_name: str,
    cache_input_dims: List[Union[int, str]],
    batch_size: int,
    num_attention_heads: int,
) -> Tuple[ModelProto, List[Union[int, str]], str, str]:
    """
    Reshapes the input and output of a kv cache in the model, so that the dimensions
    `batch_size` and `num_attention_heads` are multiplied together.

    Transform:
    ```
    |      cache_input_name
    |            |
    |           ...
    |            |
    |      cache_output_name
    ```
    to:
    ```
    |      cache_input_name
    |            |
    |      cache_input_name_reshaped
    |            |
    |           ...
    |            |
    |      cache_output_name_reshaped
    |            |
    |      cache_output_name


    :param model: The model to update
    :param cache_input_name: The name of the input to the submodel
    :param cache_output_name: The name of the output from the submodel
    :param cache_input_dims: The dimensions of the input to the submodel
    :param batch_size: The batch size of the model
    :param num_attention_heads: The number of attention heads in the model
    :return: The updated model, the updated input dimensions,
        the updated input name, and the updated output name
    """

    cache_input_name_reshaped = f"{cache_input_name}_reshaped"
    cache_output_name_reshaped = f"{cache_output_name}_reshaped"

    reshape_in_initializer_name = f"reshape_in.{cache_input_name}"
    reshape_out_initializer_name = f"reshape_out.{cache_output_name}"

    reshape_input_dims_in = copy.deepcopy(cache_input_dims)
    reshape_input_dims_out = copy.deepcopy(cache_input_dims)

    # "squash" the batch_size and num_attention_heads dimensions together
    reshape_input_dims_in[0] = batch_size * num_attention_heads
    reshape_input_dims_in.remove(num_attention_heads)

    reshape_in_array = numpy.array(
        [dim if isinstance(dim, int) else -1 for dim in reshape_input_dims_in],
        dtype=numpy.int64,
    )
    reshape_out_array = numpy.array(
        [dim if isinstance(dim, int) else -1 for dim in reshape_input_dims_out],
        dtype=numpy.int64,
    )

    reshape_in_initializer = numpy_helper.from_array(
        numpy.array(
            reshape_in_array,
            dtype=numpy.int64,
        ),
        reshape_in_initializer_name,
    )

    reshape_out_initializer = numpy_helper.from_array(
        numpy.array(
            reshape_out_array,
            dtype=numpy.int64,
        ),
        reshape_out_initializer_name,
    )

    reshape_node_in = onnx.helper.make_node(
        op_type="Reshape",
        inputs=[cache_input_name, reshape_in_initializer_name],
        outputs=[cache_input_name_reshaped],
        name=f"reshape.{cache_input_name}",
    )

    reshape_node_out = onnx.helper.make_node(
        op_type="Reshape",
        inputs=[cache_output_name_reshaped, reshape_out_initializer_name],
        outputs=[cache_output_name],
        name=f"reshape.{cache_output_name}",
    )
    graph = ONNXGraph(model)

    graph.add_node(reshape_node_in)
    graph.add_node(reshape_node_out)

    model.graph.initializer.extend([reshape_in_initializer, reshape_out_initializer])

    return (
        model,
        reshape_input_dims_in,
        cache_input_name_reshaped,
        cache_output_name_reshaped,
    )


def _find_key_value_matmul_pairs(
    graph: ONNXGraph,
) -> List[Tuple[NodeProto, NodeProto]]:
    # Find pairs of "key" and "value" MatMuls.
    # Each attention block contains a pair of MatMuls:
    #    - key MatMul that computes Q x K^T
    #    - value MatMul that computes Softmax(Q x K^T) x V
    # The function returns:
    #   [(key_matmul_0, value_matmul_0), (key_matmul_1, value_matmul_1), ...]

    key_value_matmul_pairs = []
    value_matmuls = [node for node in graph.nodes if _is_value_matmul(node, graph)]
    value_matmul_names = {node.name for node in value_matmuls}

    # for every value matmul, find the corresponding key matmul
    for value_matmul in value_matmuls:
        key_matmul = _find_key_matmul_from_value_matmul(
            value_matmul, graph, value_matmul_names
        )
        if key_matmul is not None:
            key_value_matmul_pairs.append((key_matmul, value_matmul))
        else:
            raise RuntimeError(
                f"Could not find key matmul for value matmul {value_matmul.name}"
            )

    return key_value_matmul_pairs


def _is_value_matmul(node: NodeProto, graph: ONNXGraph) -> bool:
    # A valid value MatMul needs to meet the following criteria:
    #   - is_matmul(node) is True
    #   - have no parameters
    #   - have a single parent node: a Softmax

    if not is_matmul(node) or _is_parameterized_matmul(node, graph):
        # not a matmul or MatMul op has a parameter
        return False

    for parent in graph.get_node_parents(node):
        if not isinstance(parent, NodeProto):
            continue
        if parent.op_type == "Softmax":
            # a parent is a Softmax node, assume this is a "value" MatMul
            return True

    # no parents are a softmax node
    return False


def _find_key_matmul_from_value_matmul(
    value_matmul: NodeProto,
    graph: ONNXGraph,
    value_matmul_names: Set[str],
) -> Optional[NodeProto]:
    # Perform a BFS up the model DAG from the "value" MatMul until
    # we find the corresponding "key" MatMul.
    # The "key" MatMul is assumed to be the first non-parameterized
    # MatMul we reach during the search.
    # We return None if no such matmul is found, or there is an indication that
    # we have traversed outside the self attention module (found another
    # "value" MatMul)

    seen_node_names = {value_matmul.name}
    node_queue = [value_matmul]

    while node_queue:
        current_node = node_queue.pop(0)
        node_parents = graph.get_node_parents(current_node)

        if (
            is_matmul(current_node)
            and (current_node.name != value_matmul.name)
            and not _is_parameterized_matmul(current_node, graph)
        ):
            # treat root node as regular, non MatMul node
            if current_node.name in value_matmul_names:
                _LOGGER.info(
                    f"First MatMul node found for value matmul {value_matmul.name} "
                    f"was another value matmul {current_node.name}",
                )
                return None
            else:
                # Success case -
                # first found matmul is non-parameterized
                return current_node

        for parent in node_parents:
            if not isinstance(parent, NodeProto):
                continue
            if parent.name not in seen_node_names:
                seen_node_names.add(parent.name)
                node_queue.append(parent)

    # No MatMul matched before bottoming
    _LOGGER.info(
        f"No key matmul found for value matmul {value_matmul.name}",
    )
    return None


def _value_input_idx(value_matmul: NodeProto, model: ModelProto) -> int:
    graph = ONNXGraph(model)
    # get idx of matmul that the value node is an input of
    if len(value_matmul.input) != 2:
        raise ValueError(
            f"Expected value matmul to have 2 inputs, got {len(value_matmul.input)}"
        )
    softmax_input_idx = 0  # default to softmax being on left hand side
    for idx, parent in enumerate(graph.get_node_parents(value_matmul)):
        if isinstance(parent, NodeProto) and parent.op_type == "Softmax":
            softmax_input_idx = idx
            break
    return 1 - softmax_input_idx  # return index that isn't the softmax


def is_matmul(node: NodeProto) -> bool:
    # matches against FP32 or INT8 matmul types
    return node.op_type in ["MatMul", "MatMulInteger", "Gemm", "QLinearMatMul"]


def _is_parameterized_matmul(node: NodeProto, graph: ONNXGraph) -> bool:
    # returns True if any matrix input to the node is a parameter
    # (initializer) of the graph

    # QLinearMatMul has the A,B matrices in different indices
    matrix_indices = (0, 1) if node.op_type != "QLinearMatMul" else (0, 3)

    for idx in matrix_indices:
        if graph.get_init_by_name(node.input[idx]):
            return True  # matrix input is a model weight
    return False


def _use_uint8_if_quantized(graph: ONNXGraph) -> bool:
    use_uint8_if_quantized = True  # default to True
    quantize_nodes = [node for node in graph.nodes if node.op_type == "QuantizeLinear"]
    if quantize_nodes:
        zero_point_example = graph.get_init_by_name(quantize_nodes[0].input[2])
        if zero_point_example and zero_point_example.data_type == (TensorProto.INT8):
            # quantize node exists and has INT8 input
            use_uint8_if_quantized = False
    return use_uint8_if_quantized


def _set_attention_mask_to_dynamic(model: ModelProto) -> ModelProto:
    # set the attention mask to be of the dynamic shape
    attention_mask_input = [
        input.name for input in model.graph.input if input.name == "attention_mask"
    ]
    if not attention_mask_input:
        raise ValueError("Could not find `attention_mask` input in model")
    if len(attention_mask_input) > 1:
        raise ValueError(
            "Found multiple `attention_mask` inputs in model, expected only one"
        )

    model.graph.input[1].type.tensor_type.shape.dim[
        1
    ].dim_param = "past_sequence_len + 1"
    return model
