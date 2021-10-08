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

from sparseml.sparsification.model_info import ModelInfo, ModelResult


__all__ = [
    "Analyzer",
]


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
        self._result = self._initialize_result()  # type: ModelResult

    @staticmethod
    @abstractmethod
    def available(model_info: ModelInfo, **kwargs) -> bool:
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

    def run(self, **kwargs):
        self._run(**kwargs)
        self._model_info.add_analysis_result(self._result)

    @abstractmethod
    def _initialize_result(self) -> ModelResult:
        # sets the initial ModelResult object for this analysis
        # such as analysis_type, layer selection, and result value initialization
        raise NotImplementedError()

    @abstractmethod
    def _run(self, **kwargs):
        # runs the analysis and updates self._result
        raise NotImplementedError()
