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
Code related to monitoring, analyzing, and reporting info for models in ONNX.
Records things like FLOPS, input and output shapes, kernel shapes, etc.
"""

import json
from typing import Any, Dict, List, Union

import numpy
from onnx import ModelProto

from sparseml.onnx.optim.sensitivity_pruning import pruning_loss_sens_approx
from sparseml.onnx.utils import (
    NodeShape,
    calculate_flops,
    check_load_model,
    extract_node_id,
    extract_node_shapes,
    get_kernel_shape,
    get_node_attributes,
    get_node_inputs,
    get_node_outputs,
    get_node_params,
    is_prunable_node,
)
from sparseml.utils import clean_path, create_parent_dirs


__all__ = ["NodeAnalyzer", "ModelAnalyzer"]


class NodeAnalyzer(object):
    """
    Analyzer instance for an individual node in a model

    :param model: the loaded onnx.ModelProto,
        can also be set to None if a node's kwargs are supplied
    :param node: the individual node in model,
        can also be set to None if a node's kwargs are supplied
    :param node_shape: the node's NodeShape object
    :param kwargs: additional kwargs to pass to the node
    """

    def __init__(
        self,
        model: Union[ModelProto, None],
        node: Union[Any, None],
        node_shape: Union[NodeShape, None] = None,
        **kwargs,
    ):
        if model is None and node is None:
            self._id = kwargs["id"]
            self._op_type = kwargs["op_type"]
            self._input_names = kwargs["input_names"]
            self._output_names = kwargs["output_names"]
            self._input_shapes = kwargs["input_shapes"]
            self._output_shapes = kwargs["output_shapes"]
            self._params = kwargs["params"]
            self._prunable = kwargs["prunable"]
            self._prunable_params_zeroed = kwargs["prunable_params_zeroed"]
            self._weight_name = kwargs["weight_name"]
            self._weight_shape = kwargs["weight_shape"]
            self._bias_name = kwargs["bias_name"]
            self._bias_shape = kwargs["bias_shape"]
            self._attributes = kwargs["attributes"]
            self._flops = kwargs["flops"]
            self._prunable_equation_sensitivity = (
                kwargs["prunable_equation_sensitivity"]
                if "prunable_equation_sensitivity" in kwargs
                else None
            )

            return

        if model is None or node is None:
            raise ValueError("both model and node must not be None")

        self._id = extract_node_id(node)
        self._op_type = node.op_type
        self._input_names = get_node_inputs(model, node)
        self._output_names = get_node_outputs(model, node)

        if node_shape is None:
            self._input_shapes = None
            self._output_shapes = None
        else:
            self._input_shapes = node_shape.input_shapes
            self._output_shapes = node_shape.output_shapes

        self._params = 0
        self._prunable = is_prunable_node(model, node)
        self._prunable_params = 0
        self._prunable_params_zeroed = 0
        self._weight_name = None
        self._weight_shape = None
        self._bias_name = None
        self._bias_shape = None
        self._attributes = get_node_attributes(node)

        if self._prunable:
            weight, bias = get_node_params(model, node)
            self._params += weight.val.size
            self._prunable_params += weight.val.size
            self._prunable_params_zeroed += weight.val.size - numpy.count_nonzero(
                weight.val
            )
            self._weight_name = weight.name
            self._weight_shape = [s for s in weight.val.shape]

            if bias is not None:
                self._bias_name = bias.name
                self._params += bias.val.size
                self._bias_shape = [s for s in bias.val.shape]

        kernel_shape = get_kernel_shape(self._attributes)
        self._flops = calculate_flops(
            self._op_type,
            input_shape=self._input_shapes,
            output_shape=self._output_shapes,
            weight_shape=self._weight_shape,
            kernel_shape=kernel_shape,
            bias_shape=self._bias_shape,
            attributes=self._attributes,
        )

        self._prunable_equation_sensitivity = (
            pruning_loss_sens_approx(
                self._input_shapes,
                self._output_shapes,
                self._params,
                apply_shape_change_mult=True,
            )
            if self._prunable
            else None
        )

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.dict())

    @property
    def id_(self) -> str:
        """
        :return: id of the onnx node (first output id)
        """
        return self._id

    @property
    def op_type(self) -> str:
        """
        :return: the operator type for the onnx node
        """
        return self._op_type

    @property
    def input_names(self) -> List[str]:
        """
        :return: the names of the inputs to the node
        """
        return self._input_names

    @property
    def output_names(self) -> List[str]:
        """
        :return: the names of the outputs to the node
        """
        return self._output_names

    @property
    def input_shapes(self) -> List[List[int]]:
        """
        :return: shapes for the inputs to the node
        """
        return self._input_shapes

    @property
    def output_shapes(self) -> List[List[int]]:
        """
        :return: shapes for the outputs to the node
        """
        return self._output_shapes

    @property
    def params(self) -> int:
        """
        :return: number of params in the node
        """
        return self._params

    @property
    def prunable(self) -> bool:
        """
        :return: True if the node is prunable (conv, gemm, etc), False otherwise
        """
        return self._prunable

    @property
    def prunable_params(self) -> int:
        """
        :return: number of prunable params in the node
        """
        if not self.prunable:
            return -1

        return numpy.prod(self.weight_shape).item()

    @property
    def prunable_params_zeroed(self) -> int:
        """
        :return: number of prunable params set to zero in the node
        """
        return self._prunable_params_zeroed

    @property
    def prunable_equation_sensitivity(self) -> Union[None, float]:
        """
        :return: approximated sensitivity for the layer towards pruning
            based on the layer structure and params
        """
        return self._prunable_equation_sensitivity

    @property
    def flops(self) -> Union[float, None]:
        """
        :return: number of flops to run the node
        """
        return self._flops

    @property
    def weight_name(self) -> str:
        """
        :return: the name of the weight for the node if applicable
        """
        return self._weight_name

    @property
    def weight_shape(self) -> List[int]:
        """
        :return: the shape of the weight for the node if applicable
        """
        return self._weight_shape

    @property
    def bias_name(self) -> str:
        """
        :return: name of the bias for the node if applicable
        """
        return self._bias_name

    @property
    def bias_shape(self) -> List[int]:
        """
        :return: the shape of the bias for the node if applicable
        """
        return self._bias_shape

    @property
    def attributes(self) -> Dict[str, Any]:
        """
        :return: any extra attributes for the node such as padding, stride, etc
        """
        return self._attributes

    def dict(self) -> Dict[str, Any]:
        """
        :return: dictionary representation of the current instance
        """

        return {
            "id": self.id_,
            "op_type": self.op_type,
            "input_names": self.input_names,
            "output_names": self.output_names,
            "input_shapes": self.input_shapes,
            "output_shapes": self.output_shapes,
            "params": self.params,
            "prunable": self.prunable,
            "prunable_params": self.prunable_params,
            "prunable_params_zeroed": self.prunable_params_zeroed,
            "prunable_equation_sensitivity": self.prunable_equation_sensitivity,
            "flops": self.flops,
            "weight_name": self.weight_name,
            "weight_shape": self.weight_shape,
            "bias_name": self.bias_name,
            "bias_shape": self.bias_shape,
            "attributes": self.attributes,
        }

    def __eq__(self, other: Any):
        """
        :param other: a node analyzer
        :return: True iff other is an instance of NodeAnalyzer
            and the dictionary representiations are equal.
        """
        if isinstance(other, NodeAnalyzer):
            return other.dict() == self.dict()
        else:
            return False


class ModelAnalyzer(object):
    """
    Analyze a model to get the information for every node in the model
    including params, prunable, flops, etc

    :param model: the path to the ONNX model file or the loaded onnx.ModelProto,
        can also be set to None if nodes are supplied
    :param nodes: the analyzed nodes to create the analyzer with,
        generally None and model should be passed to create a new one
    """

    @staticmethod
    def load_json(path: str):
        """
        :param path: the path to load a previous analysis from
        :return: the ModelAnalyzer instance from the json
        """
        path = clean_path(path)

        with open(path, "r") as file:
            objs = json.load(file)

        return ModelAnalyzer.from_dict(objs)

    @staticmethod
    def from_dict(dictionary: Dict[str, Any]):
        """
        :param dictionary: the dictionary to create an analysis object from
        :return: the ModelAnalyzer instance created from the dictionary
        """
        nodes = []

        for res_obj in dictionary["nodes"]:
            nodes.append(NodeAnalyzer(model=None, node=None, **res_obj))

        return ModelAnalyzer(None, nodes)

    def __init__(
        self, model: Union[ModelProto, str, None], nodes: List[NodeAnalyzer] = None
    ):
        if model is None and nodes is None:
            raise ValueError("model or nodes must not be None")

        if model is not None and nodes is not None:
            raise ValueError("model or nodes must be None, both cannot be passed")

        if model is not None:
            model = check_load_model(model)
            node_shapes = extract_node_shapes(model)
            self._nodes = [
                NodeAnalyzer(
                    model, node, node_shape=node_shapes.get(extract_node_id(node))
                )
                for node in model.graph.node
            ]
        else:
            self._nodes = nodes

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.dict())

    @property
    def nodes(self) -> List[NodeAnalyzer]:
        """
        :return: list of analyzers for each node in the model graph
        """
        return self._nodes

    def get_node(self, id_: str) -> Union[None, NodeAnalyzer]:
        """
        Get the NodeAnalyzer or the node matching the given id

        :param id_: the id to get a node for
        :return: the NodeAnalyzer that matches the id, if not found None
        """
        for node in self.nodes:
            if node.id_ == id_:
                return node

        return None

    def dict(self) -> Dict[str, Any]:
        """
        :return: dictionary representation of the current instance
        """
        return {"nodes": [node.dict() for node in self.nodes]}

    def save_json(self, path: str):
        """
        :param path: the path to save the json file at representing the analyzed
            results
        """
        if not path.endswith(".json"):
            path += ".json"

        path = clean_path(path)
        create_parent_dirs(path)

        with open(path, "w") as file:
            dictionary = self.dict()
            json.dump(dictionary, file, indent=2)

    def __eq__(self, other: Any):
        """
        :param other: a model analyzer
        :return: True iff other is an instance of ModelAnalyzer
            and the dictionary representiations of each node are equal.
        """
        if isinstance(other, ModelAnalyzer):
            return sorted(self.nodes, key=lambda node: node.id_) == sorted(
                other.nodes, key=lambda node: node.id_
            )
        return False
