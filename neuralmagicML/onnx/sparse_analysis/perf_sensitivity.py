import json
import logging
import os
from functools import reduce
from typing import Any, Dict, List, Union

import numpy as np
from neuralmagicML.onnx.utils import (
    ModelRunner,
    are_array_equal,
    get_array_from_data,
    is_prunable,
)
from tqdm import tqdm

try:
    from neuralmagic.model import benchmark_model
except (OSError, ModuleNotFoundError):
    logging.warning(
        "Installation of neuralmagic not found. Some features will be disabled."
    )
    benchmark_model = None


__all__ = ["SparsePerformanceSensitivity", "PerfAnalysisParser"]

BASELINE = "0.0"

DEFAULT_PERF_SPARSITY_LEVELS = [
    None,
    0.4,
    0.6,
    0.7,
    0.8,
    0.85,
    0.875,
    0.9,
    0.925,
    0.95,
    0.975,
    0.99,
]


class SparsePerformanceSensitivity(ModelRunner):
    def __init__(
        self,
        model_path: str,
        sparsity_levels: List = None,
        optimization_level: int = 0,
        num_cores: int = 4,
        num_warmup_iterations: int = 5,
        num_iterations: int = 30,
    ):
        if benchmark_model is None:
            raise Exception("neuralmagic must be installed to use this feature.")

        if sparsity_levels is None:
            sparsity_levels = DEFAULT_PERF_SPARSITY_LEVELS

        self.model_path = model_path
        self.sparsity_levels = sparsity_levels
        self.perf_report = {}

        self.optimization_level = optimization_level
        self.num_cores = num_cores
        self.num_warmup_iterations = num_warmup_iterations
        self.num_iterations = num_iterations

    def run(self, inputs: List, prunable_nodes: List) -> List:
        logging.debug("Running benchmarking for sparse performance sensitivity")
        batch_size = len(inputs[0])
        perf_report = {}
        for sparsity_level in tqdm(self.sparsity_levels):
            if sparsity_level == 0:
                sparsity_level = None
            key = str(sparsity_level) if sparsity_level else "0.0"
            if sparsity_level is not None:
                sparsity_level = 1 - sparsity_level
            perf_report[key] = benchmark_model(
                self.model_path,
                inputs,
                imposed_ks=sparsity_level,
                batch_size=batch_size,
                optimization_level=self.optimization_level,
                num_cores=self.num_cores,
                num_warmup_iterations=self.num_warmup_iterations,
                num_iterations=self.num_iterations,
            )

        logging.debug(
            "Finished running benchmarking for sparse performance sensitivity"
        )
        self.perf_report = PerfAnalysisParser(perf_report).get_perf_info(prunable_nodes)
        return self.perf_report

    def save(self, perf_file: str):
        with open(perf_file, "w+") as js:
            js.write(json.dumps(self.perf_report))


class PerfAnalysisParser:
    def __init__(self, sparse_data: Dict[str, Dict[str, Any]]):
        self._analysis = {}
        for sparsity in sparse_data:
            sparsity_level = float(sparsity) if sparsity != "None" else 0.0
            self._analysis[str(sparsity_level)] = PerfAnalysisAtSparsity(
                sparse_data[sparsity], sparsity_level
            )

    def get_perf_info(self, prunable_nodes: List) -> List[Dict]:
        logging.debug("Parsing sparse performance analysis")
        layer_index_to_node = self._analysis[BASELINE].match(prunable_nodes)

        perf_infos = [
            {
                "baseline": perf_info,
                "id": layer_index_to_node[index].node_key,
                "sparse": [],
                "name": layer_index_to_node[index].node_name,
            }
            for index, perf_info in enumerate(self._analysis[BASELINE].perf_info)
            if index in layer_index_to_node
        ]
        for sparsity_level in self._analysis.keys():
            if sparsity_level == BASELINE:
                continue
            for index, perf_info in enumerate(self._analysis[sparsity_level].perf_info):
                perf_infos[index]["sparse"].append(perf_info)
        logging.debug("Finished parsing sparse performance analysis")
        return perf_infos


class PerfAnalysisAtSparsity:
    def __init__(self, data: Dict[str, Any], sparsity_level: float):
        self._sparsity_level = sparsity_level
        self._layers = []
        for layer in data["layer_info"]:
            if is_prunable(layer["name"]):
                self._layers.append(PerfNode(layer))

    @property
    def perf_info(self) -> List[Dict]:
        return [
            {
                "flops": layer.flops,
                "sparsity": self._sparsity_level,
                "timing": layer.timing,
            }
            for layer in self._layers
        ]

    def match(self, prunable_nodes: List):
        name_to_idx = {}
        matched_nodes = set()
        for node_index, prunable_node in enumerate(prunable_nodes):
            name_to_idx[prunable_node.node_name] = node_index

        layer_index_to_node = {}

        for layer_index, layer in enumerate(self._layers):

            matches = [node for node in prunable_nodes if layer.is_match(node)]

            if len(matches) == 0:
                raise Exception("Unable to find match.")

            matches = sorted(
                matches,
                key=lambda match: abs(layer_index - name_to_idx[match.node_name]),
            )

            match = matches[0]

            if match.node_name in matched_nodes:
                raise Exception("Duplicate match occured.")
            matched_nodes.add(match)

            layer_index_to_node[layer_index] = match

        return layer_index_to_node


class PerfNode:
    def __init__(self, layer: Dict[str, Any]):
        self._name = layer["name"]
        self._input = get_array_from_data(layer["input_dims"])
        self._output = get_array_from_data(layer["output_dims"])
        self._kernel_shape = (
            get_array_from_data(layer["kernel_dims"]) if "kernel_dims" in layer else []
        )
        self._input_channels = layer["input_dims"]["channels"]
        self._output_channels = layer["output_dims"]["channels"]
        self._strides = get_array_from_data(layer["strides"])
        self._timing = layer["average_run_time_in_ms"]
        self._required_flops = layer["required_flops"]

    def is_match(self, prunable_node) -> bool:
        return (
            are_array_equal(self._kernel_shape, prunable_node.kernel_shape)
            and self._input_channels == prunable_node.input_channels
            and self._output_channels == prunable_node.output_channels
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def timing(self) -> float:
        return self._timing

    @property
    def flops(self) -> float:
        return self._required_flops
