import json
import logging
from functools import reduce
from typing import Any, Callable, Dict, List, Union

import numpy as np
from neuralmagicML.onnx.utils import LossRunner, ModelRunner, ORTModelRunner
from neuralmagicML.recal import KSLossSensitivityAnalysis
from scipy.special import kl_div
from tqdm import auto

DEFAULT_LOSS_SPARSITY_LEVELS = [
    0,
    0.05,
    0.2,
    0.4,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    0.975,
    0.99,
]

__all__ = [
    "OneShotKSLossSensitivity",
    "LossAnalysisParser",
]


def _get_minimums(outputs: List):
    flattened_baseline_outputs = [
        np.array([sub_output.flatten() for sub_output in output]).flatten()
        for output in outputs
    ]

    return reduce(
        lambda minimum, current: np.minimum(minimum, current),
        flattened_baseline_outputs,
    )


def _make_kl_with_min(baseline_mins: List):
    def kl_divergence(prediction, expected):
        prediction_flat = reduce(
            lambda accum, current: np.append(accum, [arr.flatten() for arr in current]),
            prediction,
            np.array([]),
        )
        expected_flat = reduce(
            lambda accum, current: np.append(accum, [arr.flatten() for arr in current]),
            expected,
            np.array([]),
        )
        ones = np.ones_like(prediction_flat)
        expected_flat += ones - baseline_mins
        prediction_flat += ones - baseline_mins

        expected_flat = np.maximum(expected_flat, ones)
        prediction_flat = np.maximum(prediction_flat, ones)

        out = np.mean(kl_div(prediction_flat, expected_flat))
        return out

    return kl_divergence


class OneShotKSLossSensitivity:
    def __init__(
        self,
        prunable_nodes: List[str],
        model: Any,
        inputs: List,
        sparsity_levels: List[float] = None,
        loss_function: Callable = None,
    ):
        if sparsity_levels is None:
            self.sparsity_levels = DEFAULT_LOSS_SPARSITY_LEVELS
        else:
            self.sparsity_levels = sparsity_levels

        self.inputs = inputs
        self.analysis = KSLossSensitivityAnalysis()
        self.analysis_parsed = {}
        model_runner = ORTModelRunner(model)

        self.baseline_outputs = [
            output["output"] for output in model_runner.run(inputs)
        ]

        self.baseline_mins = _get_minimums(self.baseline_outputs)

        if loss_function is None:
            loss_function = _make_kl_with_min(self.baseline_mins)
        self.loss_function = loss_function

    def run(
        self, inputs: List, prunable_nodes: List, model_generator: Callable,
    ):
        key_to_names = {}
        for node in prunable_nodes:
            key_to_names[node.node_key] = node.node_name
        logging.debug("Running one shot KS loss sensitivity")
        bar = auto.tqdm(
            total=len(prunable_nodes) * len(self.sparsity_levels),
            desc="KS Loss Sensitivity Analysis",
        )
        for layer_index, current_node in enumerate(prunable_nodes):
            sparsity_losses = []

            for sparsity_index, sparsity_level in enumerate(self.sparsity_levels):
                bar.update(1)
                new_model = model_generator(current_node, sparsity_level)
                self.loss_runner = LossRunner(
                    new_model, self.loss_function, ORTModelRunner
                )
                loss = np.mean(
                    [
                        output["loss"]
                        for output in self.loss_runner.run(
                            inputs, self.baseline_outputs
                        )
                    ]
                )
                sparsity_losses.append((sparsity_level, float(loss)))

            self.analysis.add_result(
                current_node.node_key, "weight", sparsity_losses,
            )
        self.analysis_parsed = LossAnalysisParser(
            self.analysis.dict(), key_to_names
        ).get_loss_info()
        logging.debug("Finished running one shot KS loss sensitivity")
        return self.analysis_parsed

    def save(self, loss_file: str):
        with open(loss_file, "w+") as js:
            js.write(json.dumps(self.analysis_parsed))


class LossAnalysisParser:
    def __init__(self, loss_content: Dict, key_to_names: Dict[str, str]):
        self._analysis = []

        for layer in loss_content["results"]:
            display_name = key_to_names[layer["param"]]
            self._analysis.append(LossNode(layer, display_name))

    def get_loss_info(self) -> List[Dict[str, Any]]:
        logging.debug("Parsing sparse loss analysis")
        loss_info = [analysis.loss_info for analysis in self._analysis]
        logging.debug("Finished parsing sparse loss analysis")
        return loss_info


class LossNode:
    def __init__(self, data: Dict[str, Any], display_name: str):
        self._id = data["param"]
        self._display_name = display_name
        self._measurements = data["sparse_measurements"]

    @staticmethod
    def measurement_to_loss(measurement):
        return {"sparsity": measurement[0], "loss": measurement[1]}

    @property
    def loss_info(self) -> Dict[str, Any]:
        sparsity_losses = [
            LossNode.measurement_to_loss(sparsity_measurement)
            for sparsity_measurement in self._measurements
        ]

        baseline_loss = [
            sparsity_loss
            for sparsity_loss in sparsity_losses
            if sparsity_loss["sparsity"] == 0
        ]

        baseline_loss = baseline_loss[0] if len(baseline_loss) > 0 else None
        return {
            "id": self._id,
            "name": self._display_name,
            "baseline": baseline_loss,
            "sparse": [
                sparsity_loss
                for sparsity_loss in sparsity_losses
                if sparsity_loss["sparsity"] != 0
            ],
        }
