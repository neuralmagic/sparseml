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
Code for running analysis on neural networks
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, Optional, Set, Tuple

import numpy
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from sparseml.optim import default_pruning_sparsities_loss
from sparseml.sparsification.model_info import (
    ModelInfo,
    ModelResult,
    PruningSensitivityResult,
    PruningSensitivityResultTypes,
)


__all__ = [
    "AnalyzerProgress",
    "Analyzer",
    "PruningLossSensitivityMagnitudeAnalyzer",
]


class AnalyzerProgress(BaseModel):
    """
    Simple class for tracking model analyzer progress
    """

    step: int = Field(
        title="step",
        description="current step of the Analyzer",
    )
    total_steps: int = Field(
        title="total_steps",
        description="total steps Analyzer will run",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        title="metadata",
        default=None,
        description=(
            "optional metadata dict of property names to values for current "
            "analysis step"
        ),
    )

    @property
    def progress(self) -> float:
        """
        :return: float progress on [0,1] scale
        """
        return float(self.step) / float(self.total_steps)


class Analyzer(ABC):
    """
    Base abstract class for model analyzers. Analyzers should be able to detect
    if given a ModelInfo object and other keyword inputs if they should run their
    analysis.

    :param model_info: ModelInfo object of the model to be analyzed. after
        running this analysis, the analysis_results of this ModelInfo object
        will be updated
    """

    def __init__(self, model_info: ModelInfo):
        self._model_info = model_info
        self.result = self._initialize_result()  # type: ModelResult

    @classmethod
    @abstractmethod
    def available(cls, model_info: ModelInfo, **kwargs) -> bool:
        """
        Abstract method that subclasses must implement to determine if
        given the model info and keyword arguments that the Analyzer can
        run its analysis

        :param model_info: ModelInfo object of the model to be analyzed
        :param kwargs: additional keyword arguments that will be passed to the run
            function
        :return: True if given the inputs, this analyzer can run its analysis. False
            otherwise
        """
        raise NotImplementedError()

    def run(self, show_progress: bool = False, **kwargs) -> ModelResult:
        """
        Runs the given analysis by calling to the underlying run_iter method
        :param show_progress: set True to display a tqdm progress bar. default is False
        :param kwargs: key word arguments validated by available() to run this analysis
        :return: the final result from this analysis. this result will also be
            added to the ModelInfo object of this Analyzer
        """
        bar = None
        prev_progress = 0

        for progress, _ in self.run_iter(**kwargs):
            if show_progress and bar is None:
                bar = tqdm(
                    total=progress.total_steps,
                    desc=f"{self.result.analysis_type} Analyzer Progress",
                )

            if bar is not None:
                bar.update(progress.step - prev_progress)
                prev_progress = progress.step

        if bar is not None:
            bar.close()

        return self.result

    def run_iter(
        self,
        **kwargs,
    ) -> Generator[Tuple[AnalyzerProgress, ModelResult], None, None]:
        """
        runs the analysis stepwise using the abstract _run_iter method yielding an
        AnalyzerProgress and the in progress ModelResult at each step

        After the last step, the final results will be added to the given ModelInfo

        :param kwargs: key word arguments validated by available() to run this analysis
        """
        for progress, result in self._run_iter(**kwargs):
            yield progress, result

        self._model_info.add_analysis_result(self.result)

    @abstractmethod
    def _initialize_result(self) -> ModelResult:
        # sets the initial ModelResult object for this analysis
        # such as analysis_type, layer selection, and result value initialization
        raise NotImplementedError()

    @abstractmethod
    def _run_iter(
        self,
        **kwargs,
    ) -> Generator[Tuple[AnalyzerProgress, ModelResult], None, None]:
        # runs the analysis and updates self.result
        raise NotImplementedError()


class PruningLossSensitivityMagnitudeAnalyzer(Analyzer, ABC):
    """
    Base class for running pruning loss sensitivity weight magnitude analysis.
    A valid in-framework model with prunable weights is required to run this analysis

    pruning_loss_analysis_sparsity_levels is an optional run argument to set the
    sparsities that this analysis will run at. if not set, the value defaults to
    sparsml.optim.default_pruning_sparsities_loss(extended=True)
    """

    @classmethod
    def available(cls, model_info: ModelInfo, **kwargs) -> bool:
        """
        Determines if given the available kwargs and ModelInfo, that weight magnitude
        analysis is available. `model` must exist in the given keyword arguments and
        be a valid model of the given framework and include all prunable parameters
        named in the ModelInfo

        :param model_info: ModelInfo object of the model to be analyzed
        :param kwargs: keyword arguments that will be passed in to this analysis. model
            must be included for this analysis to be available
        :return: True if given the inputs, this analyzer can run its analysis. False
            otherwise
        """
        if "model" not in kwargs:
            return False
        return cls.validate_model(
            model_info.get_prunable_param_names(), kwargs["model"]
        )

    @staticmethod
    @abstractmethod
    def validate_model(prunable_param_names: Set[str], model: Any) -> bool:
        """
        Validates that all prunable parameter names in the ModelInfo layer_info
        exist in the given model and that the given model is of the correct framework

        :param prunable_param_names: set of prunable parameter names found in the model
            info
        :param model: model to validate
        :return: True if this is a valid model for weight mangitude pruning analysis.
            False otherwise
        """
        raise NotImplementedError()

    @abstractmethod
    def get_named_prunable_params(self, model: Any) -> Dict[str, numpy.ndarray]:
        """
        loads the prunable parameters in a standardized way so that weight magnitude
        analysis may be run on each

        :param model: model to load the prunable parameters from
        :return: dictionary of prunable parameter name as listed in the ModelInfo to
            a numpy array of the values of the parameter
        """
        raise NotImplementedError()

    def _initialize_result(self) -> PruningSensitivityResult:
        return PruningSensitivityResult(PruningSensitivityResultTypes.LOSS)

    def _run_iter(
        self,
        **kwargs,
    ) -> Generator[Tuple[AnalyzerProgress, PruningSensitivityResult], None, None]:
        named_params = self.get_named_prunable_params(kwargs["model"])
        num_params = len(named_params)
        sparsity_levels = (
            kwargs["pruning_loss_analysis_sparsity_levels"]
            if "pruning_loss_analysis_sparsity_levels" in kwargs
            else default_pruning_sparsities_loss(True)
        )

        for idx, (name, param) in enumerate(named_params.items()):
            yield AnalyzerProgress(step=idx, total_steps=num_params), self.result

            sorted_param_vals = numpy.sort(numpy.abs(param.flatten()))
            prev_sparsity_idx = 0

            for sparsity in sparsity_levels:
                sparsity_idx = round(sparsity * sorted_param_vals.size)
                if sparsity_idx >= len(sorted_param_vals):
                    sparsity_idx = len(sorted_param_vals) - 1

                if sparsity <= 1e-9:
                    sparsity = 0.0
                    sparse_avg = 0.0
                else:
                    if sparsity_idx > prev_sparsity_idx:
                        sparse_avg = (
                            sorted_param_vals[prev_sparsity_idx:sparsity_idx]
                            .mean()
                            .item()
                        )
                        prev_sparsity_idx = sparsity_idx + 1

                self.result.add_layer_sparsity_result(name, sparsity, sparse_avg)

        yield AnalyzerProgress(step=num_params, total_steps=num_params), self.result
