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

import collections
from typing import Dict, List, Optional, Tuple

import numpy as np
from tensorflow.python.framework import tensor_util
from toposort import toposort

from sparseml.optim import AnalyzedLayerDesc
from sparseml.tensorflow_v1.utils.helpers import tf_compat
from sparseml.tensorflow_v1.utils.variable import get_op_input_var


__all__ = ["analyze_module"]


def analyze_module(
    session: Optional[tf_compat.Session],
    graph: Optional[tf_compat.Graph],
    op_names: Optional[List[str]] = None,
    op_types: Optional[List[str]] = None,
):
    """
    Analyze a module at certain layers

    :param session: running session encapsulating the analyzed module
    :param graph: graph of the module; if None then the session is required,
        and the encapsulated graph is to be analyzed
    :param op_names: list of names of layers to be analyzed;
        if None then all layers are analyzed for an aggregated result
    :param op_types: the operation types that will be analyzed, default (Conv2D, MatMul)
    :return: the analyzed layer descriptions or the module description if no op_names
    """
    if op_types is None:
        op_types = ["Conv2D", "MatMul"]

    _validate(session, graph)
    ops = [
        o
        for o in graph.get_operations()
        if (o.type in op_types) and (op_names is None or o.name in op_names)
    ]
    ops_desc = _analyze_ops(session, graph, ops)  # Dict[str, AnalyzedLayerDesc]
    return ops_desc


def _validate(session: tf_compat.Session, graph: tf_compat.Graph):
    """
    Check and make sure the session and graph are consistent.
    Provided session and graph might be reassigned for consistency.

    :param session: Current session
    :param graph: Current graph
    """
    if not session and not graph:
        raise ValueError("Either session or graph must be provided")
    if session:
        if graph != tf_compat.get_default_graph():
            raise ValueError("Inconsistent session and graph")
        graph = tf_compat.get_default_graph()
    else:
        session = tf_compat.Session(graph=graph)


def _analyze_ops(
    session: tf_compat.Session, graph: tf_compat.Graph, ops: List[tf_compat.Operation]
) -> Dict[str, AnalyzedLayerDesc]:
    """
    Analyze operations for their properties

    :param session: Current session
    :graph: Current graph
    :ops: List of operations in the graph to be analyzed

    :return A dictionary of AnalyzedLayerDesc object for each operation name
    """
    exec_orders = _op_exec_order(graph)
    ops_desc = {}
    for op in ops:
        assert type(op) == tf_compat.Operation
        desc = AnalyzedLayerDesc(op.name, op.type)
        desc.params = _count_parameters(session, op)
        desc.zeroed_params = _count_parameters(session, op, "zeroed")
        desc.prunable_params = _count_parameters(session, op, "prunable")
        desc.params_dims = _get_parameters_dims(op)
        desc.prunable_params_dims = _get_parameters_dims(op)
        desc.execution_order = exec_orders[op.name]
        desc.input_shape = tuple(
            [tuple(_from_tensor_shape(ten.shape)) for ten in op.inputs]
        )
        desc.output_shape = tuple(
            [tuple(_from_tensor_shape(ten.shape)) for ten in op.outputs]
        )
        ops_desc[op.name] = desc

    op_flops = _profile_flops(graph, ops)
    for op in ops:
        ops_desc[op.name].flops = -1  # Unused
        ops_desc[op.name].total_flops = op_flops[op.name]

    return ops_desc


def _profile_flops(
    graph: tf_compat.Graph, ops: List[tf_compat.Operation]
) -> Dict[str, int]:
    """
    Using TF Profiling to get FLOPS of operations

    :param graph: Current graph
    :param ops: List of operations

    :return A dictionary of FLOPS for each operation name
    """
    gdef = graph.as_graph_def()
    new_graph = tf_compat.Graph()

    with new_graph.as_default():
        # Modify the graph to work around a bug before running tf.profile
        # https://github.com/tensorflow/tensorflow/issues/20960
        import_prefix_name = "NM_IMPORT"
        _replace_incomplete_shape_placeholers(
            gdef, import_prefix_name=import_prefix_name
        )
        opt = (
            tf_compat.profiler.ProfileOptionBuilder(
                tf_compat.profiler.ProfileOptionBuilder.float_operation()
            )
            .with_node_names(show_name_regexes=[".*Conv2D.*", ".*MatMul.*"])
            .with_empty_output()
            .build()
        )
        prof_stats = tf_compat.profiler.profile(new_graph, options=opt)
        op_names = {"{}/{}".format(import_prefix_name, o.name): o.name for o in ops}
        op_flops = {
            op_names[child.name]: child.total_float_ops
            for child in prof_stats.children
            if child.name in list(op_names.keys())
        }
        return op_flops


def _count_parameters(
    session: tf_compat.Session, op: tf_compat.Operation, parameter_type: str = "all"
) -> int:
    """
    Count the number of parameters of input weight tensor of an operation

    :param session: Current session
    :param op: An operation
    :param parameter_type: Type of parameters to count

    :return Number of parameters
    """
    assert parameter_type in {"all", "zeroed", "prunable"}
    n_params = None
    # For both MatMul and Conv2D we assume the parameters will be
    # the last one of the two inputs
    weight_tensor = get_op_input_var(op)
    n_params = int(np.prod(weight_tensor.shape.as_list()))
    if parameter_type == "zeroed":
        tensor_vals = session.run(weight_tensor)
        nonzeros = np.count_nonzero(tensor_vals)
        n_params -= nonzeros
    return n_params


def _get_parameters_dims(op: tf_compat.Operation) -> Tuple[int, ...]:
    """
    Get the dimensions of an operation

    :param op: An operation
    :return List of dimensions
    """
    weight_tensor = get_op_input_var(op)
    return tuple(weight_tensor.shape.as_list())


def _from_tensor_shape(shape: tf_compat.TensorShape) -> List[int]:
    """
    Convert from TensorShape with potentially incomplete to list.
    Incomplete dimension is encoded by -1

    :param shape: Given shape

    :return List of elements along dimensions
    """
    new_shape = [-1 if d.value is None else d.value for d in shape]
    return new_shape


def _replace_incomplete_shape_placeholers(gdef, import_prefix_name="NM_IMPORT"):
    """
    Replace placeholders of incomplete shapes with new ones with incomplete dimensions
    being replaced by size of 1

    :param: gdef Graph definition
    :param import_prefix_name Prefix used for the resulting graph
    """
    placeholders = [o for o in gdef.node if o.op == "Placeholder"]
    input_map = {}
    for pl in placeholders:
        dtype = tf_compat.as_dtype(pl.attr["dtype"].type)
        shape = tensor_util.TensorShapeProtoToList(pl.attr["shape"].shape)
        new_shape = [1 if d == -1 else d for d in shape]
        new_pl = tf_compat.placeholder(
            dtype, shape=new_shape, name="new_{}".format(pl.name)
        )
        input_map[pl.name] = new_pl

    # Get correct import_graph_def function for TF version
    import_graph_def = (
        tf_compat.graph_util.import_graph_def
        if hasattr(tf_compat.graph_util, "import_graph_def")
        else tf_compat.import_graph_def
    )
    import_graph_def(gdef, input_map=input_map, name=import_prefix_name)


def _op_exec_order(g: tf_compat.Graph):
    """
    Get execution order of operations in a graph

    :param g: Current graph

    :return Dictionary of execution order in integer for each operation name
    """
    deps = collections.defaultdict(set)
    for op in g.get_operations():
        deps[op.name] = set([inp.name for inp in op.inputs])
        for out in op.outputs:
            deps[out.name].add(op.name)
    order = {}
    ordered_name_sets = list(toposort(deps))
    for idx, name_set in enumerate(ordered_name_sets):
        for name in name_set:
            order[name] = idx
    return order
