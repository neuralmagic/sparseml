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

"""
Code to assist in the compression of structured-pruned models
"""

from typing import Dict, List, Set, Union

import onnx

from sparseml.onnx.utils import ONNXGraph, get_node_attributes


__all__ = ["get_param_structured_pruning_group_dependencies"]


_PRUNABLE_OP_TYPES = ["Conv", "Gemm", "MatMul"]
_OUTPUT_CHANNEL_OP_TYPES = _PRUNABLE_OP_TYPES + ["BatchNormalization"]


def get_param_structured_pruning_group_dependencies(
    model: Union[onnx.ModelProto, str],
    structure_type: str = "filter",
) -> Dict[str, List[str]]:
    """
    :param model: model to generate pruning groups and dependencies for
    :param structure_type: valid options are 'filter' and 'channel'. Generates
        dependency map for corresponding pruning scheme. Default is 'filter'
    :return: dictionary of parameter names that should be grouped during
        structured pruning to a list of parameter names whose parameters should
        be updated accordingly to the param group pruning results. prunable parameter
        names will be represented as a comma separated string
    """
    if structure_type not in ["filter", "channel"]:
        raise ValueError(
            f"invalid structure_type {structure_type}. not in ['filter', 'channel']"
        )

    if isinstance(model, str):
        model = onnx.load(model)

    graph = ONNXGraph(model)
    param_name_to_dependents = {}  # Dict[str, Set[str]]
    for node in model.graph.node:
        if node.op_type not in _PRUNABLE_OP_TYPES or (
            graph.get_init_by_name(node.input[1]) is None
        ):
            # main param not found or not prunable
            continue

        param_name_to_dependents[node.input[1]] = _get_node_dependency_names(
            graph, node, structure_type
        )

    # merge disjoint sets of dependencies (could improve with union-find)
    prunable_param_group_to_dep_params = []  # List[Tuple[List, Set]]
    for prunable_param_name, dep_params in param_name_to_dependents.items():
        intersected_group_idxs = {
            idx
            for idx, (_, group_dep_params) in enumerate(
                prunable_param_group_to_dep_params
            )
            if not dep_params.isdisjoint(group_dep_params)
        }
        new_group_val = ([prunable_param_name], dep_params)
        if not intersected_group_idxs:
            prunable_param_group_to_dep_params.append(new_group_val)
        else:
            non_intersected_vals = []
            for idx, (prunable_param_group, group_dep_params) in enumerate(
                prunable_param_group_to_dep_params
            ):
                if idx not in intersected_group_idxs:
                    non_intersected_vals.append(
                        (prunable_param_group, group_dep_params)
                    )
                else:
                    new_group_val = (
                        new_group_val[0] + prunable_param_group,
                        new_group_val[1].union(group_dep_params),
                    )
            prunable_param_group_to_dep_params = non_intersected_vals + [new_group_val]

    return {
        ",".join(prunable_param_group): list(dependent_params)
        for prunable_param_group, dependent_params in prunable_param_group_to_dep_params
    }


def _get_next_layer_deps(
    graph: ONNXGraph, node: onnx.NodeProto, structure_type: str
) -> List[onnx.NodeProto]:
    return (
        [
            parent_node
            for parent_node in graph.get_node_parents(node)
            if isinstance(parent_node, onnx.NodeProto)
        ]
        if structure_type == "channel"
        else graph.get_node_children(node)
    )


def _get_node_output_ids(nodes: List[onnx.NodeProto]) -> Set[str]:
    if isinstance(nodes, onnx.NodeProto):
        nodes = [nodes]
    ids = set()
    for node in nodes:
        ids.update(set(node.output))
    return ids


def _get_node_dependency_names(
    graph: ONNXGraph, node: onnx.NodeProto, structure_type: str
) -> Set[str]:
    # returns a list of parameters whose should be pruned to match
    # the target dimensions of this node
    unchecked_nodes = _get_next_layer_deps(graph, node, structure_type)
    seen_output_ids = _get_node_output_ids(unchecked_nodes)
    dependent_params = set()

    if structure_type == "filter" and len(node.input) > 2:
        # node bias depends on num filters
        dependent_params.add(node.input[2])

    while unchecked_nodes:
        current_node = unchecked_nodes.pop(0)
        if not isinstance(current_node, onnx.NodeProto):
            continue

        if current_node.op_type in _OUTPUT_CHANNEL_OP_TYPES:
            prunable = current_node.op_type in _PRUNABLE_OP_TYPES
            params = (
                list(current_node.input[1:])  # skip layer input tensor
                if not (prunable and structure_type != "filter")
                else [current_node.input[1]]  # bias not dependent on prev filter
            )

            for param in params:
                if graph.get_init_by_name(param) is not None:
                    dependent_params.add(param)
            if prunable and not _is_group_conv(current_node):
                # continue on other branches, do not go past prunable nodes
                continue
        dep_nodes = _get_next_layer_deps(graph, current_node, structure_type)
        for dep_node in dep_nodes:
            dep_node_ids = _get_node_output_ids(dep_node)
            if dep_node_ids.isdisjoint(seen_output_ids):
                unchecked_nodes.append(dep_node)
                seen_output_ids.update(dep_node_ids)

    return dependent_params


def _is_group_conv(node: onnx.NodeProto) -> bool:
    if not node.op_type == "Conv":
        return False
    attrs = get_node_attributes(node)
    groups = attrs.get("group", 1)
    try:
        return int(groups) != 1
    except Exception:
        return False
