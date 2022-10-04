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
Helper functions to edit ONNX Graphs.
"""

from collections import defaultdict
from typing import Iterable, List, Optional, Union

import numpy
import onnx
from onnx import ModelProto, NodeProto, TensorProto, numpy_helper
from toposort import toposort_flatten

from sparseml.onnx.utils.helpers import get_node_params


__all__ = [
    "ONNXGraph",
    "update_model_param",
    "swap_node_output",
    "remove_node_and_params_from_graph",
    "override_model_batch_size",
    "prune_unstructured",
    "prune_model_one_shot",
    "prune_model_one_shot_iter",
]


class ONNXGraph(object):
    """
    Class for quick look-up of ONNX graph nodes and initializers. If graph state
    changes outside of ONNXGraph class functions, update() should be called.

    :param model: the ONNX graph to represent
    """

    def __init__(self, model: ModelProto):
        self._model = model
        self._output_id_to_node = {}
        self._input_id_to_nodes = defaultdict(list)
        self._name_to_initializer = {}

        self.update()

    @property
    def nodes(self) -> Iterable[NodeProto]:
        """
        :return: ordered collection of nodes in this graph
        """
        return self._model.graph.node

    def update(self, model: Optional[ModelProto] = None):
        """
        Update the graph state based on the model this graph represents or
        the given model.

        :param model: model to represent. defaults to current loaded model state
        """
        self._model = model or self._model

        # nodes
        self._output_id_to_node = {}
        self._input_id_to_nodes = defaultdict(list)
        for node in self._model.graph.node:
            self._store_node_edges(node)

        # initializers
        self._name_to_initializer = {
            init.name: init for init in self._model.graph.initializer
        }

    def get_init_by_name(
        self,
        name: str,
        allow_optional: bool = True,
    ) -> Optional[TensorProto]:
        """
        :param name: name of initializer
        :param allow_optional: if True and the given name is not found as an
            initializer, None will be returned. Otherwise a KeyError will be raised
        :return: tensor of initializer with given name, returns None if the name does
            not exist in the cached graph
        """
        init = self._name_to_initializer.get(name, None)
        if not allow_optional and init is None:
            raise KeyError(f"Unable to find initializer {name} in ONNX model")
        return init

    def get_node_by_output_id(self, id: str) -> Optional[TensorProto]:
        """
        :param id: name of output id of node
        :return: the associated node if it is present in the graph, None otherwise
        """
        return self._output_id_to_node.get(id)

    def get_node_parents(
        self, node: NodeProto
    ) -> List[Union[NodeProto, TensorProto, None]]:
        """
        :param node: node to get the input objects for
        :return: input nodes or tensors of this node in order. if an input does not
            exist, None will be returned in its place
        """
        inputs = []
        for input_id in node.input:
            inp = None
            if input_id in self._output_id_to_node:
                inp = self._output_id_to_node[input_id]
            elif input_id in self._name_to_initializer:
                inp = self._name_to_initializer[input_id]
            inputs.append(inp)
        return inputs

    def get_node_single_parent(
        self, node: NodeProto, index: int
    ) -> Union[NodeProto, None]:
        """
        :param node: the node to get the parent node of
        :param index: choose which input to search
        :return: parent of node if it only has one parent, otherwise None
        """
        input_id = node.input[index]
        if input_id not in self._output_id_to_node:
            return None
        return self._output_id_to_node[input_id]

    def get_node_children(self, node: NodeProto) -> List[NodeProto]:
        """
        :param node: the node to get the children node of
        :return: list of nodes that include this node as an output
        """
        children = []
        for output_id in node.output:
            children.extend(self._input_id_to_nodes[output_id])
        return children

    def get_node_single_child(self, node: NodeProto) -> Union[NodeProto, None]:
        """
        :param node: the node to get the child node of
        :return: child of node if it only has one child, otherwise None
        """
        children = self.get_node_children(node)
        return children[0] if len(children) == 1 else None

    def add_node(self, node: NodeProto):
        """
        Adds the given node to the model and graph state

        :param node: node to add to the model
        """
        self._model.graph.node.append(node)
        self._store_node_edges(node)

    def update_node_input(
        self, node: NodeProto, input_id: str, input_idx: Optional[int] = None
    ):
        """
        :param node: node to update the inputs of
        :param input_id: new input_id to attach to the node
        :param input_idx: optional index of the node input list to update,
            if none is given, the new input id will be appended to the input list
        """
        if input_idx is not None:
            if node in self._input_id_to_nodes[node.input[input_idx]]:
                self._input_id_to_nodes[node.input[input_idx]].remove(node)
            node.input[input_idx] = input_id
        else:
            node.input.append(input_id)
        self._input_id_to_nodes[input_id].append(node)

    def delete_node(self, node: NodeProto):
        """
        deletes the given node from the graph

        :param node: node to delete
        """
        self._model.graph.node.remove(node)
        self._delete_node_edges(node)

    def delete_nodes(self, nodes: List[NodeProto]):
        """
        deletes the given nodes from the graph
        :param nodes: list of nodes to delete
        """
        node_ouptut_ids_to_delete = {node.output[0] for node in nodes}
        nodes_to_keep = []
        for node in self._model.graph.node:
            if node.output[0] in node_ouptut_ids_to_delete:
                self._delete_node_edges(node)
            else:
                nodes_to_keep.append(node)
        self._model.graph.ClearField("node")
        self._model.graph.node.extend(nodes_to_keep)

    def delete_initializers(self, initializers: List[Union[str, TensorProto]]):
        """
        deletes the given initializers from the model

        :param initializers: list of initializers or initializer names to delete
        """
        inits_to_delete = {
            init if isinstance(init, str) else init.name for init in initializers
        }
        inits_to_keep = []
        for init in self._model.graph.initializer:
            if init.name in inits_to_delete:
                # keep edge reference if nodes in the graph still point to the
                # initializer name
                if not self._input_id_to_nodes[init.name]:
                    del self._input_id_to_nodes[init.name]
                del self._name_to_initializer[init.name]
            else:
                inits_to_keep.append(init)
        self._model.graph.ClearField("initializer")
        self._model.graph.initializer.extend(inits_to_keep)

    def delete_unused_initializers(self):
        """
        deletes tensors in the initializer list that are not listed as inputs to any
        node in the current graph state or directly passed as model outputs
        """
        output_names = {out.name for out in self._model.graph.output}
        self.delete_initializers(
            [
                init
                for init in self._model.graph.initializer
                if not self._input_id_to_nodes[init.name]
                and (init.name not in output_names)
            ]
        )  # delete inits that have no edge

    def sort_nodes_topologically(self):
        """
        Sorts the order of the graph Node repeated field in place in topological
        order as per the ONNX Model proto specifications
        """
        # build toposort DAG input and sort
        model_dag = defaultdict(set)  # node_id -> dependencies
        for parent_node_id, child_nodes in self._input_id_to_nodes.items():
            if parent_node_id not in self._output_id_to_node:
                continue  # parent is an initializer, not node
            # standardize all references to nodes by their first output id
            parent_node_id = self._output_id_to_node[parent_node_id].output[0]
            for child_node in child_nodes:
                model_dag[child_node.output[0]].add(parent_node_id)
        sorted_node_ids = toposort_flatten(model_dag)

        # deduplicate any nodes from the sorted list
        updated_node_list = []
        seen_ids = set()
        for node_id in sorted_node_ids:
            if node_id in seen_ids:
                continue  # a node could have multiple ids, all ids will be updated
            node = self._output_id_to_node[node_id]
            updated_node_list.append(node)
            seen_ids.update(node.output)

        # update model node list with topo sorted list
        assert len(updated_node_list) == len(self._model.graph.node)
        self._model.graph.ClearField("node")
        self._model.graph.node.extend(updated_node_list)

    def _store_node_edges(self, node: NodeProto):
        for output_id in node.output:
            self._output_id_to_node[output_id] = node
        for input_id in node.input:
            self._input_id_to_nodes[input_id].append(node)

    def _delete_node_edges(self, node: NodeProto):
        # remove node edges from cache
        for output_id in node.output:
            del self._output_id_to_node[output_id]
        for input_id in node.input:
            self._input_id_to_nodes[input_id].remove(node)


def update_model_param(
    model: ModelProto,
    param_name: str,
    val: numpy.ndarray,
) -> None:
    """
    Removes the parameter with name param_name from the model
    Creates a new parameter using val
    Adds val to the model with name param_name as an update

    :param model: The model to update
    :param param_name: The parameter name in the model to update
    :param val: The new value of the parameter
    """
    param_matches = [
        param for param in model.graph.initializer if param.name == param_name
    ]
    if param_matches:
        model.graph.initializer.remove(param_matches[0])
    new_param = numpy_helper.from_array(val, param_name)
    model.graph.initializer.append(new_param)


def swap_node_output(node: onnx.NodeProto, output: str) -> None:
    """
    Deletes the current output of the node and replaces it with the provided value
    Assumes that the node only has one output

    :param node: Node to change the output of
    :param output: New output value
    """
    node.output.pop()
    node.output.append(output)


def remove_node_and_params_from_graph(model: ModelProto, node: onnx.NodeProto) -> None:
    """
    Deletes a node from the mdoel graph as well as its parameters listed in node.input

    :param model: Model to delete from
    :param node: Node to delete
    """
    model.graph.node.remove(node)
    graph = ONNXGraph(model)
    graph.delete_unused_initializers()


def _override_tensor_batch_dim(model, tensor, batch_size):
    for init in model.graph.initializer:
        if init.name == tensor.name:
            # This tensor is actually an initializer => skip
            return

    shape = tensor.type.tensor_type.shape

    # skip tensors with variable batch sizes
    if not shape.dim[0].dim_param and shape.dim[0].dim_value > 0:
        shape.dim[0].dim_value = batch_size


def override_model_batch_size(model: ModelProto, batch_size: int) -> ModelProto:
    """
    Rewrites any positive batch dimensions in the model inputs or outputs to the
    given batch_size

    :param model: Model to modify
    :param batch_size: Batch size to enforce
    :return: the given model with inputs and outputs set to batch_size if the batch
        dimensions are not -1.
    """
    for tensor in model.graph.input:
        # This may not work for ONNX graphs that have hard-coded reshape nodes
        _override_tensor_batch_dim(model, tensor, batch_size)
    # Do the same for outputs
    for tensor in model.graph.output:
        # Ignore augmented _Reduce nodes
        if "_Reduce" not in tensor.name:
            _override_tensor_batch_dim(model, tensor, batch_size)


def prune_unstructured(array: numpy.ndarray, sparsity: float) -> numpy.ndarray:
    """
    Prune a numpy array with unstructured sparsity according to magnitude pruning

    :param array: the array to prune (introduce zeros), will remove the lowest
        absolute values in the array
    :param sparsity: the sparsity value, as a decimal, to impose in the array
    :return: the pruned array
    """
    array = numpy.array(array)  # make a copy because arrays from onnx are read only
    sparse_index = int(round(sparsity * array.size) - 1)

    if sparse_index < 0:
        return array

    sorted_array = numpy.sort(numpy.abs(array.flatten()))
    sparse_thresh = sorted_array[sparse_index]
    array[numpy.abs(array) < sparse_thresh] = 0

    return array


def prune_model_one_shot(
    model: ModelProto, nodes: List[NodeProto], sparsity: Union[float, List[float]]
):
    """
    Prune a model in-place with one shot pruning (no retraining) according to
    magnitude pruning. Does so in an unstructured way currently

    :param model: the model to apply pruning to
    :param nodes: the nodes within the model to prune to the desired sparsities
    :param sparsity: the sparsity level to prune all nodes to if a float,
        or the sparsity level to prune each node to if a list of floats
    :return: the new, pruned model
    """
    if not isinstance(sparsity, Iterable):
        tmp = float(sparsity)
        sparsity = [tmp for _ in range(len(nodes))]

    if len(nodes) != len(sparsity):
        raise ValueError(
            "len(nodes) {} does not match len(sparsity) {}".format(
                len(nodes), len(sparsity)
            )
        )

    for node, sparsity in zip(nodes, sparsity):
        weight, bias = get_node_params(model, node)
        pruned_weight_val = prune_unstructured(weight.val, sparsity)
        update_model_param(model, weight.name, pruned_weight_val)


def prune_model_one_shot_iter(
    model: ModelProto, nodes: List[NodeProto], sparsity: Union[float, List[float]]
):
    """
    Iteratively prune a model in-place with one shot pruning (no retraining) according
    to magnitude pruning. Does so in an unstructured way currently

    :param model: the model to apply pruning to
    :param nodes: the nodes within the model to prune to the desired sparsities
    :param sparsity: the sparsity level to prune all nodes to if a float,
        or the sparsity level to prune each node to if a list of floats
    """
    if not isinstance(sparsity, Iterable):
        tmp = float(sparsity)
        sparsity = [tmp for _ in range(len(nodes))]

    if len(nodes) != len(sparsity):
        raise ValueError(
            "len(nodes) {} does not match len(sparsity) {}".format(
                len(nodes), len(sparsity)
            )
        )

    for index, (node, sparsity) in enumerate(zip(nodes, sparsity)):
        weight, bias = get_node_params(model, node)
        pruned_weight_val = prune_unstructured(weight.val, sparsity)
        update_model_param(model, weight.name, pruned_weight_val)
        yield (index + 1) / len(nodes)
