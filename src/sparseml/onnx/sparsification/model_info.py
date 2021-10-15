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
Classes for describing models and layers in ONNX models.
"""

from collections import OrderedDict
from typing import Optional, Union

import numpy
import onnx
from onnx import ModelProto, NodeProto, numpy_helper

from sparseml.onnx.utils import ONNXGraph, get_node_attributes
from sparseml.sparsification import LayerInfo
from sparseml.sparsification import ModelInfo as BaseModelInfo


__all__ = [
    "ONNXModelInfo",
]


class ONNXModelInfo(BaseModelInfo):
    """
    Class for extracting and serializing ONNX model metadata, layers info, and
    analysis results

    :param model: ONNX ModelProto object, or path to ONNX model file
    """

    def extract_layer_info(self, model: ModelProto) -> "OrderedDict[str, LayerInfo]":
        """
        :param model: ONNX model to extract LayerInfo of
        :return: ordered dictionary of layer name to LayerInfo object for the prunable
            model layers
        """
        layers = OrderedDict()
        graph = ONNXGraph(model)
        graph.sort_nodes_topologically()  # for execution order

        for node in graph.nodes:
            layer_info = None
            if node.op_type == "Conv":
                layer_info = self._make_conv_layer_info(node, graph, len(layers))
            elif node.op_type == "Gemm":
                layer_info = self._make_gemm_layer_info(node, graph, len(layers))
            elif node.op_type == "MatMul":
                layer_info = self._make_matmul_layer_info(node, graph, len(layers))

            if layer_info is not None:
                layers[layer_info.name] = layer_info

        return layers

    @staticmethod
    def _make_conv_layer_info(
        node: NodeProto,
        graph: ONNXGraph,
        execution_order: int,
    ) -> Optional[NodeProto]:
        param = ONNXModelInfo._get_node_param_array(node, graph)
        if param is None:
            return

        attributes = get_node_attributes(node)
        kernel_shape = attributes.get("kernel_shape", list(param.shape[2:]))
        groups = attributes.get("group", 1)
        stride = attributes.get("strides", [1] * (len(param.shape) - 2))
        padding = attributes.get("pads", [0, 0] * (len(param.shape) - 2))

        return LayerInfo.conv_layer(
            name=node.input[1],
            in_channels=param.shape[1] * groups,
            out_channels=param.shape[0],
            kernel_shape=kernel_shape,
            bias=len(node.input) > 2,
            groups=groups,
            stride=stride,
            padding=padding,
            execution_order=execution_order,
            attributes=dict(sparsity=_param_sparsity(param)),
        )

    @staticmethod
    def _make_gemm_layer_info(
        node: NodeProto,
        graph: ONNXGraph,
        execution_order: int,
    ) -> Optional[NodeProto]:
        param = ONNXModelInfo._get_node_param_array(node, graph)
        if param is None:
            return

        attributes = get_node_attributes(node)
        if attributes.get("transB", 0) != 0:
            # ensure that param shape is (in_channels, out_channels)
            param = param.transpose()

        return LayerInfo.linear_layer(
            name=node.input[1],
            in_channels=param.shape[0],
            out_channels=param.shape[-1],
            bias=len(node.input) > 2,
            execution_order=execution_order,
            attributes=dict(sparsity=_param_sparsity(param)),
        )

    @staticmethod
    def _make_matmul_layer_info(
        node: NodeProto,
        graph: ONNXGraph,
        execution_order: int,
    ) -> Optional[NodeProto]:
        param = ONNXModelInfo._get_node_param_array(node, graph)
        if param is None:
            return

        return LayerInfo.linear_layer(
            name=node.input[1],
            in_channels=param.shape[0],
            out_channels=param.shape[-1],
            bias=False,
            execution_order=execution_order,
            attributes=dict(sparsity=_param_sparsity(param)),
        )

    @staticmethod
    def _get_node_param_array(
        node: NodeProto,
        graph: ONNXGraph,
        param_idx: int = 1,
    ) -> Optional[numpy.ndarray]:
        if len(node.input) <= param_idx:
            # no such param exists
            return None
        param = graph.get_init_by_name(node.input[1])
        if param is None:
            # input is not a param stored as an initializer in the graph
            return None
        return numpy_helper.to_array(param)

    def _validate_model(self, model: Union[ModelProto, str]):
        # validate model type and load from file if necessary
        if not isinstance(model, (ModelProto, str)):
            raise ValueError(
                f"{self.__class__.__name__} must be instantiated with a ModelProto "
                f"object or string file path to one. Received: {type(model)}"
            )
        if isinstance(model, str):
            model = onnx.load(model)
        return model


def _param_sparsity(param: numpy.ndarray) -> float:
    # return param sparsity rounded to 4 decimal places
    return float(param.size - numpy.count_nonzero(param)) / float(param.size)
