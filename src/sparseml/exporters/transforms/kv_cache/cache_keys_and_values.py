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

ALLOWED_NODES_BEFORE_SOFTMAX = ["Cast", "Reshape", "QuantizeLinear"]
ALLOWED_NODES_FOLLOWING_CONCAT = ["Transpose", "QuantizeLinear"]
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
    4.  For the key or value cache, if there is a Transpose node present before
        the MatMul, the concatenation will be performed before the Transpose node
    6. (Optional) To account for the variance in the operations in the vicinity
        of the "key" and "value" MatMuls, the user can specify whether to additionally
        inject a Reshape or Transpose node, so that the dimensions of the cache
        inputs/outputs are compatible with the values they are concatenated with.

    This transform also sets the subset of kv cache inputs/outputs dimensions (
    num_attention_heads and hidden_size_kv_cache) to the appropriate static values.

    :param num_attention_heads: number of attention heads of the model
    :param hidden_size_kv_cache: hidden size of the key and value cache
    :param multiply_batch_by_num_att_heads: every model created by
        this transformation, will have kv_cache inputs/outputs that have dimensions:
        [`batch_size`,`num_attention_heads`,`past_sequence_len`,`hidden_size_kv_cache`]

        However, internally, there may be a need of reshaping the kv_cache
        inputs/outputs ("merging" the `batch_size` and `num_attention_heads`
        dimensions) so that it is compatible with the values it is concatenated with.
        If True, the batch size will be multiplied by the number of attention
        heads just before the appropriate concatenation node (as reflected by
        the "Reshape" nodes in the diagram below).
    :param transpose_value_input: if not None, transpose the kv cache value
        input before the "value" MatMul. The argument needs to be a tuple of
        4 integers that represent the permutation of the input dimensions.
        This will insert a Transpose node before the "value" MatMul. If
        multiply_batch_by_num_att_heads is True, the Transpose node will be
        inserted before the Reshape node.
    :param transpose_key_input: works analogously to transpose_value_input,
        but for the key input.


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
    |       ...  ...
    |        |  |
    |         |
    |   "value" MatMul
    |        |
    |       ...
    ```
    to

    ```
    |
    | KeyCache
    |    |
    | Transpose
    |(optional)
    |    |
    |    |         Key
    | Reshape      |
    |(optional)   |
    |    |       |
    |     |    |
    |      | |
    |      |
    |   Concat ------------> OutputKeyCache
    |      |
    |      |     Query
    |     ...      |
    |      |      |
    |      |      |
    |      |      |       ValueCache
    |      |      |          |
    |      |      |      Transpose
    |      |      |      (optional)
    |      |     |          |
    |      |    |        Reshape
    |      |   |         (optional)
    |       |  |    Value  |
    |       | |       |   |
    |        |        |  |
    |   "key" MatMul  | |
    |        |        |
    |       ...   Concat --> OutputValueCache
    |        |      |
    |     Softmax  |
    |        |   ...
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
        multiply_batch_by_num_att_heads: bool,
        transpose_value_input: Optional[Tuple[int, int, int, int]] = None,
        transpose_key_input: Optional[Tuple[int, int, int, int]] = None,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.hidden_size_kv_cache = hidden_size_kv_cache
        self.multiply_batch_by_num_att_heads = multiply_batch_by_num_att_heads
        self.transpose_value_input = transpose_value_input
        self.transpose_key_input = transpose_key_input

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
                transpose_input=self.transpose_key_input,
                multiply_batch_by_num_att_heads=self.multiply_batch_by_num_att_heads,  # noqa E501
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
                transpose_input=self.transpose_value_input,
                multiply_batch_by_num_att_heads=self.multiply_batch_by_num_att_heads,  # noqa E501
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
    multiply_batch_by_num_att_heads: bool = True,
    transpose_input: Optional[Tuple[int, int, int, int]] = None,
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
    :param multiply_batch_by_num_att_heads: If True, the batch size of the
        kv cache is multiplied by the number of attention heads before the
        concat node.
    :param transpose_input: If not None, transpose the input to the cache
        before the concat node. If `multiply_batch_by_num_att_heads` is True,
        the transpose is applied after the batch size is multiplied by the
        number of attention heads.
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

    if (
        isinstance(cache_parent, NodeProto)
        and cache_parent.op_type in ALLOWED_NODES_FOLLOWING_CONCAT
    ):
        while cache_parent.op_type in ALLOWED_NODES_FOLLOWING_CONCAT:
            pre_cache_input_id = cache_parent.input[0]
            cache_parent = graph.get_node_single_parent(cache_parent, index=0)

    else:
        pre_cache_input_id = node.input[cache_input_idx]

    cache_input_name_concat = cache_input_name
    cache_output_name_concat = cache_output_name
    cache_input_dims_concat = CACHE_INPUT_DIMS

    if transpose_input:
        (
            graph,
            cache_input_dims_concat,
            cache_input_name_concat,
            cache_output_name_concat,
        ) = transpose_kv_cache_inputs_outputs(
            graph=graph,
            cache_input_name=cache_input_name_concat,
            cache_output_name=cache_output_name_concat,
            cache_input_dims=cache_input_dims_concat,
            transpose_input=transpose_input,
        )

    if multiply_batch_by_num_att_heads:
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

    for _node in model.graph.node:
        for input_idx, input_id in enumerate(_node.input):
            if input_id == pre_cache_input_id and _node.name != concat_node.name:
                _node.input[input_idx] = cache_output_name_concat

    if node.op_type == "MatMulInteger":
        quantize_linear = graph.get_node_single_parent(node, cache_input_idx)
        quantize_linear_parent = graph.get_node_single_parent(quantize_linear, 0)
        if quantize_linear_parent is None:
            quantize_linear_parent = concat_node

        concat_node = move_quantize_linear_node(
            quantize_linear=quantize_linear,
            quantize_linear_parent=quantize_linear_parent,
            concat=concat_node,
            cache_input_idx=cache_input_idx,
            graph=graph,
        )

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


def transpose_kv_cache_inputs_outputs(
    graph: ONNXGraph,
    cache_input_name: str,
    cache_output_name: str,
    cache_input_dims: List[Union[int, str]],
    transpose_input: Tuple[int, int, int, int],
) -> Tuple[ModelProto, List[Union[int, str]], str, str]:
    """
    Transposes the input and output of a kv cache in the model
    according to the transpose_input sequence

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
    |      cache_input_name_transposed
    |            |
    |           ...
    |            |
    |      cache_output_name_transposed
    |            |
    |      cache_output_name

    :param graph: The graph to update
    :param cache_input_name: The name of the input to the submodel
    :param cache_output_name: The name of the output from the submodel
    :param transpose_input: The permutation of the input dimensions
    :param cache_input_dims: The dimensions of the input to the submodel
    :return: The updated model, the updated input dimensions,
        the updated input name, and the updated output name
    """

    cache_input_name_transposed = f"{cache_input_name}_transposed"
    cache_output_name_transposed = f"{cache_output_name}_transposed"

    transpose_node_in = onnx.helper.make_node(
        op_type="Transpose",
        inputs=[cache_input_name],
        outputs=[cache_input_name_transposed],
        name=f"transpose.{cache_input_name}",
        perm=transpose_input,
    )
    transpose_node_out = onnx.helper.make_node(
        op_type="Transpose",
        inputs=[cache_output_name_transposed],
        outputs=[cache_output_name],
        name=f"transpose.{cache_output_name}",
        perm=transpose_input,
    )
    transposed_input_dims = [cache_input_dims[i] for i in transpose_input]

    graph.add_node(transpose_node_in)
    graph.add_node(transpose_node_out)

    return (
        graph,
        transposed_input_dims,
        cache_input_name_transposed,
        cache_output_name_transposed,
    )


def move_quantize_linear_node(
    quantize_linear: NodeProto,
    quantize_linear_parent: NodeProto,
    concat: NodeProto,
    cache_input_idx: str,
    graph: ONNXGraph,
) -> NodeProto:
    """
    Moves a QuantizeLinear node before the `concat` node, so
    that the data that arrives from `ConcatNodeParent` to `concat`
    is already quantized (see the diagram below). This is required
    so that the `concat` node joins the quantized data from the
    `ConcatNodeParent` with the quantized kv cache input.

    Transforms
    ```
    |  ConcatNodeParent
    |    |
    |    |   Key/Value Cache(uint8)
    |    |        |
    |     |    ...
    |      |    |
    |       | |
    |        |
    |     Concat
    |        |
    |       ...
    |        |
    |  QuantizeLinear
    |        |
    |        |  ...
    |        |  |
    |         |
    |    QLinearMatMul
    ```
    to

    ```
    |  ConcatNodeParent
    |   |
    | QuantizeLinear
    |   |
    |   |   Key/Value Cache (uint8)
    |    |        |
    |     |    ...
    |      |    |
    |       | |
    |        |
    |     Concat
    |        |
    |        |
    |        |  ...
    |        |  |
    |         |
    |    QLinearMatMul
    ```
    :param quantize_linear: The QuantizeLinear node to move.
        In reality, this node will be removed and a new node,
        that inherits attributes from this node, will be created
        in the proper place.
    :param quantize_linear_parent: The parent of the QuantizeLinear node.
    :param concat: The concat node to move the QuantizeLinear node before.
    :param cache_input_idx: The index of the cache input in the concat node.
    :param graph: The graph to update.
    :return: The updated Concat node.
    """
    if quantize_linear.op_type != "QuantizeLinear":
        raise ValueError(
            f"It is expected that the node: {quantize_linear.name} "
            f"has opset: QuantizeLinear, but it has op_type: {quantize_linear.ops_type}"
        )
    quantize_linear_child = graph.get_node_single_child(quantize_linear)
    if quantize_linear_child.op_type != "MatMulInteger":
        raise ValueError(
            f"It is expected that the node: {quantize_linear.name} "
            "has opset: MatMulInteger, but it has "
            f"op_type: {quantize_linear_child.op_type}"
        )

    # remove the dependency on the QuantizeLinear node from its
    # neighbouring nodes by connecting output of its parent to its child
    quantize_linear_child.input[cache_input_idx] = quantize_linear_parent.output[0]

    # get the node precedes the concat node and does not come from
    # the kv cache input. Then place the QuantizeLinear node after it
    concate_node_parent = graph.get_node_parents(concat)[1]
    quantize_linear.input[0] = concate_node_parent.output[0]
    concat.input[1] = quantize_linear.output[0]
    return concat


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
    value_matmuls = [node for node in graph.nodes if is_value_matmul(node, graph)]
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


def is_value_matmul(
    node: NodeProto,
    graph: ONNXGraph,
    allowed_nodes_before_softmax: Set[str] = ALLOWED_NODES_BEFORE_SOFTMAX,
) -> bool:
    """
    Returns True if the node is a "value" MatMul, i.e. a MatMul that meets
    the following criteria:
        -   is_matmul(node) is True
        -   node has no parameters
        -   node has a single `Softmax` parent node
                or
            the parent node `Softmax` is preceded by
            a set of nodes that are specified in the
            `allowed_nodes_before_softmax` set

    :param node: node to check
    :param graph: graph containing the node
    :param allowed_nodes_before_softmax: set of node types that are allowed
        to be located between the node in question a Softmax node, so that
        the node can still be considered a "value" MatMul
    """

    if not is_matmul(node) or _is_parameterized_matmul(node, graph):
        # not a matmul or MatMul op has a parameter
        return False

    parent = graph.get_node_single_parent(node, index=0)
    while parent.op_type in allowed_nodes_before_softmax:
        if not isinstance(parent, NodeProto):
            break
        parent = graph.get_node_single_parent(parent, index=0)
        if parent is None:
            raise ValueError(
                "While traversing the graph to find a Softmax that precedes "
                f"the candidate for a `value` MatMul: {node.name}, found a node "
                f"with multiple parents {parent.name}. "
                "It is assumed that the graph that connects the Softmax "
                "node and the `value` MatMul node is a linear chain of nodes "
                "and thus none of the encountered nodes should have multiple "
                "parents"
            )

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
    expected_num_inputs = 4 if value_matmul.op_type == "MatMulInteger" else 2

    if len(value_matmul.input) != expected_num_inputs:
        raise ValueError(
            f"Expected value matmul to have {expected_num_inputs} "
            f"inputs, got {len(value_matmul.input)}"
        )

    softmax_input_idx = 0  # default to softmax being on left hand side
    for idx, parent in enumerate(graph.get_node_parents(value_matmul)):
        if isinstance(parent, NodeProto):
            # if a parent is a softmax or the parent of value matmul is a direct
            # child of a softmax (quantized scenario), then the softmax is the
            # idx'th input to the value matmul
            if (
                parent.op_type == "Softmax"
                or graph.get_node_single_parent(parent, 0).op_type == "Softmax"
            ):
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
