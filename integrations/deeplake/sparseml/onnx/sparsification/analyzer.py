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
Analyzer class implementations for ONNX
"""


import logging
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import numpy
from onnx import ModelProto, numpy_helper

from sparseml.onnx.utils import DataLoader, DeepSparseAnalyzeModelRunner, ONNXGraph
from sparseml.optim import default_pruning_sparsities_perf
from sparseml.sparsification import Analyzer, AnalyzerProgress, ModelInfo
from sparseml.sparsification import (
    PruningLossSensitivityMagnitudeAnalyzer as BasePruningLossMagnitudeAnalyzer,
)
from sparseml.sparsification import (
    PruningSensitivityResult,
    PruningSensitivityResultTypes,
)


__all__ = [
    "PruningLossSensitivityMagnitudeAnalyzer",
    "PruningPerformanceSensitivityAnalyzer",
    "get_analyzer_impls",
]


_LOGGER = logging.getLogger(__name__)


class PruningLossSensitivityMagnitudeAnalyzer(BasePruningLossMagnitudeAnalyzer):
    """
    Class for performing weight mangitude pruning sensitivity analysis on ONNX models

    pruning_loss_analysis_sparsity_levels is an optional run argument to set the
    sparsities that this analysis will run at. if not set, the value defaults to
    sparsml.optim.default_pruning_sparsities_loss(extended=True)
    """

    @staticmethod
    def validate_model(prunable_param_names: Set[str], model: ModelProto) -> bool:
        """
        Validates that all prunable parameter names in the ModelInfo layer_info
        exist in the given model and that the given model is of the correct framework

        :param prunable_param_names: set of prunable parameter names found in the model
            info
        :param model: model to validate
        :return: True if this is a valid model for weight mangitude pruning analysis.
            False otherwise
        """
        return _validate_onnx_model_analyzer(prunable_param_names, model)

    def get_named_prunable_params(self, model: Any) -> Dict[str, numpy.ndarray]:
        """
        loads the prunable parameters in a standardized way so that weight magnitude
        analysis may be run on each

        :param model: model to load the prunable parameters from
        :return: dictionary of prunable parameter name as listed in the ModelInfo to
            a numpy array of the values of the parameter
        """
        graph = ONNXGraph(model)
        return {
            layer_name: numpy_helper.to_array(graph.get_init_by_name(layer_name, False))
            for layer_name, layer_info in self._model_info.layer_info.items()
            if layer_info.prunable
        }


class PruningPerformanceSensitivityAnalyzer(Analyzer):
    """
    Class for running pruning performance sensitivity analysis on a model against
    the DeepSparse engine. deepsparse must be installed to be available.

    pruning_perf_analysis_sparsity_levels is an optional run argument to set the
    sparisities that this analysis will run at. if not set, the value defaults to
    sparsml.optim.default_pruning_sparsities_perf()

    :param model_info: ModelInfo object of the model to be analyzed. after
        running this analysis, the analysis_results of this ModelInfo object
        will be updated
    :param batch_size: batch size to run analysis at. Default is 1
    :param num_cores: number of CPU cores to run analysis with. Default
        is all available on the system
    :param iterations_per_check: number of benchmarking iterations
        to run for each sparsity level. Default is 10
    :param warmup_iterations_per_check: number of warmup iterations
        to run at each saprsity level. Default is 5
    """

    def __init__(
        self,
        model_info: ModelInfo,
        batch_size: int = 1,
        num_cores: Optional[int] = None,
        iterations_per_check: int = 10,
        warmup_iterations_per_check: int = 5,
    ):
        self._batch_size = batch_size
        self._iterations_per_check = iterations_per_check
        self._warmup_iterations_per_check = warmup_iterations_per_check

        # try grabbing default max cores if needed; for tracking purposes
        try:
            from deepsparse.cpu import cpu_details

            self._num_cores = num_cores or cpu_details()[0]
        except Exception:
            self._num_cores = num_cores

        super().__init__(model_info)

    @classmethod
    def available(cls, model_info: ModelInfo, **kwargs) -> bool:
        """
        Determines if given the available kwargs and ModelInfo, that pruning
        performance analysis wioth deepsparse is available. `model` must exist in
        the given keyword arguments and be an onnx ModelProto with all prunable
        parameters from the ModelInfo available in its initializers list. Additionally
        deepsparse must be installed and the DeepSparseAnalyzeModelRunner must be
        available

        :param model_info: ModelInfo object of the model to be analyzed
        :param kwargs: keyword arguments that will be passed in to this analysis. model
            must be included for this analysis to be available
        :return: True if given the inputs, this analyzer can run its analysis. False
            otherwise
        """
        if "model" not in kwargs or not DeepSparseAnalyzeModelRunner.available():
            return False
        return _validate_onnx_model_analyzer(
            model_info.get_prunable_param_names(), kwargs["model"]
        )

    def _initialize_result(self) -> PruningSensitivityResult:
        return PruningSensitivityResult(
            PruningSensitivityResultTypes.PERF,
            attributes=dict(
                batch_size=self._batch_size,
                num_cores=self._num_cores,
                iterations_per_check=self._iterations_per_check,
                warmup_iterations_per_check=self._warmup_iterations_per_check,
            ),
        )

    def _run_iter(
        self,
        **kwargs,
    ) -> Generator[Tuple[AnalyzerProgress, PruningSensitivityResult], None, None]:
        sparsity_levels = (
            kwargs["pruning_perf_analysis_sparsity_levels"]
            if "pruning_perf_analysis_sparsity_levels" in kwargs
            else default_pruning_sparsities_perf()
        )
        num_steps = len(sparsity_levels)

        model = kwargs["model"]
        data_loader = DataLoader.from_model_random(model, self._batch_size, -1)

        # build map of possible layer identifiers to prunable param name
        id_to_param_name = {}
        param_names = self._model_info.get_prunable_param_names()
        for param_name in param_names:
            layer_info = self._model_info.layer_info[param_name]

            # by output id
            output_id = layer_info.attributes.get("node_output_id")
            if output_id is not None:
                id_to_param_name[output_id] = param_name

            # by node name
            node_name = layer_info.attributes.get("node_name")
            if node_name is not None:
                id_to_param_name[node_name] = param_name

            # directly match to param name
            id_to_param_name[param_name] = param_names

        runner = DeepSparseAnalyzeModelRunner(model, self._batch_size, self._num_cores)

        for idx, sparsity in enumerate(sparsity_levels):
            if sparsity <= 1e-9:
                sparsity = None  # to enforce dense execution

            yield AnalyzerProgress(step=idx, total_steps=num_steps), self.result

            results = runner.run(
                data_loader,
                show_progress=False,
                num_iterations=self._iterations_per_check,
                num_warmup_iterations=self._warmup_iterations_per_check,
                imposed_ks=sparsity,
                max_steps=1,
            )[0][0]
            _LOGGER.debug(
                "measured perf results for one shot sparsity {}".format(sparsity)
            )

            # model sparsity -> average time in seconds
            self.result.add_model_sparsity_result(
                sparsity or 0.0, results["average_total_time"] / 1000.0
            )

            for layer in results["layer_info"]:
                layer_name = id_to_param_name.get(
                    layer["canonical_name"],
                    id_to_param_name.get(layer["name"]),  # fallback to internal name
                )
                if layer_name is not None:
                    self.result.add_layer_sparsity_result(
                        layer_name,
                        sparsity if sparsity is not None else 0.0,
                        layer["average_run_time_in_ms"] / 1000.0,
                    )
        yield AnalyzerProgress(step=num_steps, total_steps=num_steps), self.result


def get_analyzer_impls() -> List[Analyzer]:
    """
    :return: list of ONNX Analyzer implementations
    """
    return [
        PruningLossSensitivityMagnitudeAnalyzer,
        PruningPerformanceSensitivityAnalyzer,
    ]


def _validate_onnx_model_analyzer(
    prunable_param_names: Set[str], model: ModelProto
) -> bool:
    if not isinstance(model, ModelProto):
        _LOGGER.debug(
            "ONNX model Analyzer expected model of type onnx.ModelProto, found: %s",
            str(type(model)),
        )
        return False
    initializer_names = {init.name for init in model.graph.initializer}
    is_valid = prunable_param_names.issubset(initializer_names)

    if not is_valid:
        _LOGGER.debug(
            "ONNX model Analyzer unable to find prunable params with names %s in "
            "model initializer list",
            ", ".join(prunable_param_names - initializer_names),
        )

    return is_valid
