import json
import logging
import math
import os
import re
from functools import reduce
from typing import Any, Dict, Iterable, List

import numpy as np
import onnx
import onnxruntime as rt
from neuralmagicML.onnx.sparse_analysis import (
    OneShotKSLossSensitivity,
    SparsePerformanceSensitivity,
)
from onnx import numpy_helper
from onnx.helper import ValueInfoProto, get_attribute_value, make_graph, make_model

__all__ = ["RecalModel"]


def _is_prunable(node) -> bool:
    return node.op_type == "Conv" or node.op_type == "Gemm"


def _get_weight_input(inputs) -> List:
    return [inp for inp in inputs if "weight" in inp.name][0]


def prepare_weights_for_sparisty(init):
    weights = numpy_helper.to_array(init)
    density = np.count_nonzero(weights) / weights.size
    sparsity = 1.0 - density

    weights_flatten = numpy_helper.to_array(init).flatten()
    weights_flatten = np.absolute(weights_flatten)
    weights_flatten.sort()

    index_interval = (
        int(len(weights_flatten) / 1000) if len(weights_flatten) >= 1000 else 1
    )
    percent_interval = index_interval / len(weights_flatten)
    iterations = math.ceil(len(weights_flatten) / index_interval)
    return weights_flatten, index_interval, percent_interval, iterations


def _node_name(node) -> str:
    return node.name if node.name else node.output[0]


class RecalModel:
    def __init__(self, path: str):
        self._path = path
        self._model = onnx.load_model(path)
        onnx.checker.check_model(self._model)

        prunable_nodes = [node for node in self._model.graph.node if _is_prunable(node)]

        # Get output shapes of prunable nodes
        output_to_node = {}
        for node in prunable_nodes:
            intermediate_layer_value_info = ValueInfoProto()
            intermediate_layer_value_info.name = node.output[0]
            output_to_node[node.output[0]] = _node_name(node)
            self._model.graph.output.append(intermediate_layer_value_info)

        sess_options = rt.SessionOptions()
        sess_options.log_severity_level = 3
        sess = rt.InferenceSession(self._model.SerializeToString(), sess_options)

        # resets model
        self._model = onnx.load_model(path)
        self._input_shapes = [inp.shape for inp in sess.get_inputs()]
        node_to_shape = reduce(
            lambda accum, current: (
                accum.update({output_to_node[current.name]: current.shape}) or accum
            )
            if current.name in output_to_node
            else accum,
            [out for out in sess.get_outputs()],
            {},
        )

        # Create Prunable Node
        input_to_nodes = {}
        for node in prunable_nodes:
            for inp in node.input:
                input_to_nodes[inp] = _node_name(node)

        node_to_inputs = reduce(
            lambda accum, node: accum.update({node: []}) or accum,
            [_node_name(node) for node in prunable_nodes],
            {},
        )

        for inp in self._model.graph.initializer:
            if inp.name not in input_to_nodes:
                continue
            node_name = input_to_nodes[inp.name]
            node_to_inputs[node_name].append(inp)

        self._prunable_nodes = [
            PrunableNode(
                node, node_to_inputs[_node_name(node)], node_to_shape[_node_name(node)]
            )
            for node in prunable_nodes
        ]

    @property
    def prunable_nodes(self) -> List:
        return self._prunable_nodes

    @property
    def prunable_layers(self) -> List[Dict]:
        return [node.layer_info for node in self._prunable_nodes]

    @property
    def sparse_analysis_loss_approx(self) -> List[Dict]:
        return [node.sparse_analysis_loss_approx for node in self._prunable_nodes]

    @property
    def sparse_analysis_perf_approx(self) -> List[Dict]:
        return [node.sparse_analysis_perf_approx for node in self._prunable_nodes]

    @property
    def input_shapes(self) -> List:
        return self._input_shapes

    @property
    def model_path(self) -> str:
        return self._path

    @property
    def onnx_model(self):
        return self._model

    def run_sparse_analysis_perf(
        self,
        perf_file: str,
        inputs: Iterable,
        sparsity_levels: List[float] = None,
        optimization_level: int = 0,
        num_cores: int = 4,
        num_warmup_iterations: int = 5,
        num_iterations: int = 30,
    ):
        try:
            inputs_data = next(inputs)
        except StopIteration:
            raise Exception(
                "No input data found. Must have at least 1 input to use one shot ks loss sensitivity"
            )

        perf_sensitivity = SparsePerformanceSensitivity(
            self.model_path,
            sparsity_levels=sparsity_levels,
            optimization_level=optimization_level,
            num_cores=num_cores,
            num_warmup_iterations=num_warmup_iterations,
            num_iterations=num_iterations,
        )
        sparse_analysis_perf = perf_sensitivity.run(inputs_data, self.prunable_nodes)
        perf_sensitivity.save(perf_file)
        return sparse_analysis_perf

    def get_model_pruned_at_node(self, current_node: Any, sparsity_level: float):
        new_weights = []
        for weight_idx, weight in enumerate(self.onnx_model.graph.initializer):
            if weight.name == current_node.node_name:
                weight = current_node.pruned_weight(sparsity_level)
            new_weights.append(weight)

        new_graph = make_graph(
            self.onnx_model.graph.node,
            self.onnx_model.graph.name,
            self.onnx_model.graph.input,
            self.onnx_model.graph.output,
            initializer=new_weights,
            value_info=self.onnx_model.graph.value_info,
        )

        return make_model(new_graph)

    def one_shot_ks_loss_sensitivity(
        self,
        loss_file: str,
        inputs: Iterable,
        sparsity_levels: List[float] = None,
        samples_per_measurement: int = 5,
    ):
        inputs_data = []
        for inp in inputs:
            inputs_data.append(inp)
            if len(inputs_data) >= samples_per_measurement:
                break

        if len(inputs_data) == 0:
            raise Exception(
                "No input data found. Must have at least 1 input to use one shot ks loss sensitivity"
            )

        ks_loss_sensitivity = OneShotKSLossSensitivity(
            [layer.node_name for layer in self.prunable_nodes],
            self.onnx_model,
            inputs_data,
            sparsity_levels,
        )

        ks_loss_sensitivity.run(
            inputs_data,
            self.prunable_nodes,
            lambda current_node, sparsity_level: self.get_model_pruned_at_node(
                current_node, sparsity_level
            ),
        )

        ks_loss_sensitivity.save(loss_file)


class PrunableNode:
    def __init__(self, node, weights: List, output_shape: List):
        self._node = node
        self._weights = weights
        self._output_shape = output_shape

        self._attributes = reduce(
            lambda accum, attribute: accum.update(
                {attribute.name: get_attribute_value(attribute)}
            )
            or accum,
            node.attribute,
            {},
        )

        attribute_string = reduce(
            lambda accum, attribute_key: f"{accum},{attribute_key}={self._attributes[attribute_key]}"
            if accum is not None
            else f"{attribute_key}={self._attributes[attribute_key]}",
            self._attributes.keys(),
            None,
        )

        self._node_key = str(node.output)

        self._input_ids = node.input

        self._weight_input = _get_weight_input(self._weights)
        self._node_name = self._weight_input.name

        self._index_sorted = np.argsort(
            np.absolute(numpy_helper.to_array(self._weight_input)), axis=None
        )

    def pruned_weight(self, sparsity_level: float):
        weight_as_np = numpy_helper.to_array(self._weight_input)
        new_weights = weight_as_np.flatten()
        max_sparse_count = math.ceil(self._index_sorted.size * sparsity_level)
        for index, ranking in enumerate(self._index_sorted):
            if ranking < max_sparse_count:
                new_weights[index] = 0
        return numpy_helper.from_array(
            new_weights.reshape(weight_as_np.shape).astype(np.float32),
            name=self.node_name,
        )

    @property
    def layer_info(self) -> Dict:
        return {
            "attributes": self._attributes,
            "id": self._node_key,
            "name": self._node_name,
            "inputs": [node_input for node_input in self._node.input],
            "output": [node_output for node_output in self._node.output],
            "op_type": self._node.op_type,
        }

    @property
    def sparse_analysis_loss_approx(self) -> Dict:
        (
            weights_flatten,
            index_interval,
            percent_interval,
            iterations,
        ) = prepare_weights_for_sparisty(self._weight_input)

        sparse = []
        cummulative_loss = 0
        for i in range(iterations):
            cummulative_loss += np.sum(
                weights_flatten[i * index_interval : (i + 1) * index_interval]
            )
            sparsity_percent = min((i + 1) * percent_interval * 100, 100)
            sparse.append({"loss": cummulative_loss, "sparsity": sparsity_percent})

        return {
            "id": self._node_key,
            "sparse": sparse,
            "baseline": {"loss": 0.0, "sparsity": 0.0},
        }

    @property
    def sparse_analysis_perf_approx(self) -> Dict:
        (
            _,
            index_interval,
            percent_interval,
            iterations,
        ) = prepare_weights_for_sparisty(self._weight_input)

        flops = 0
        bias_flops = 0
        for weights in self._weights:
            if "bias" in weights.name:
                bias_flops = np.prod(numpy_helper.to_array(weights).shape)
            elif self._node.op_type == "Gemm":
                flops += np.prod(numpy_helper.to_array(weights).shape)
            else:
                kernel = np.prod(self._attributes["kernel_shape"])
                flops += (
                    kernel
                    * numpy_helper.to_array(weights).shape[1]
                    * np.prod(self.output_shape[1:])
                )

        for weights in self._weights:
            sparse = []
            current_flops = flops

            weights_size = numpy_helper.to_array(weights).size
            index_interval = int(weights_size / 1000) if weights_size >= 1000 else 1
            percent_interval = index_interval / weights_size
            iterations = math.ceil(weights_size / index_interval)

            for i in range(iterations):
                sparsity_percent = min((i + 1) * percent_interval * 100, 100)
                current_flops = flops * (1 - sparsity_percent / 100) + bias_flops
                sparse.append(
                    {
                        "flops": float(current_flops),
                        "timing": None,
                        "sparsity": sparsity_percent,
                    }
                )

        return {
            "id": self._node_key,
            "baseline": {"flops": float(flops + bias_flops), "timing": None},
            "sparse": sparse,
        }

    @property
    def node_key(self):
        return self._node_key

    @property
    def node_name(self):
        return self._node_name

    @property
    def kernel_shape(self) -> List[int]:
        return (
            self._attributes["kernel_shape"]
            if "kernel_shape" in self._attributes
            else []
        )

    @property
    def strides(self) -> List[int]:
        return self._attributes["strides"] if "strides" in self._attributes else []

    @property
    def output_shape(self) -> List[int]:
        return self._output_shape

    @property
    def output_channels(self) -> int:
        return self._weight_input.dims[0]

    @property
    def input_channels(self) -> int:
        return self._weight_input.dims[1]

    @property
    def weights(self) -> List:
        return self._weights

    @property
    def op_type(self) -> str:
        return self._node.op_type

    @property
    def contains_input(self) -> bool:
        return "input" in self._input_ids
