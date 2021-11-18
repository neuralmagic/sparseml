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
Base classes for describing models and layers in ML framework neural networks.
"""


import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

import numpy
from pydantic import BaseModel, Field, root_validator

from sparseml.utils import clean_path, create_parent_dirs


__all__ = [
    "LayerInfo",
    "Result",
    "ModelResult",
    "PruningSensitivityResult",
    "PruningSensitivityResultTypes",
    "ModelInfo",
]


class LayerInfo(BaseModel):
    """
    Class for storing properties about a layer in a model
    """

    name: str = Field(
        title="name",
        description="unique name of the layer within its model",
    )
    op_type: str = Field(
        title="op_type",
        description="type of layer, i.e. 'conv', 'linear'",
    )
    params: Optional[int] = Field(
        title="params",
        default=None,
        description=(
            "number of non-bias parameters in the layer. must be included "
            "for prunable layers"
        ),
    )
    bias_params: Optional[int] = Field(
        title="bias_params",
        default=None,
        description="number of bias parameters in the layer",
    )
    prunable: bool = Field(
        title="prunable",
        default=False,
        description="True if the layers non-bias parameters can be pruned",
    )
    flops: Optional[int] = Field(
        title="flops",
        default=None,
        description="number of float operations within the layer",
    )
    execution_order: int = Field(
        title="execution_order",
        default=-1,
        description="execution order of the layer within the model",
    )
    attributes: Optional[Dict[str, Any]] = Field(
        title="attributes",
        default=None,
        description="dictionary of string attribute names to their values",
    )

    @root_validator(pre=True)
    def check_params_if_prunable(_, values):
        prunable = values.get("prunable")
        params = values.get("params")
        if prunable and not params:
            raise ValueError(
                f"Prunable layers must have non 0 number of params given {params} "
                f"for layer {values.get('name')} with prunable set to {prunable}"
            )
        return values

    @classmethod
    def linear_layer(
        cls, name: str, in_channels: int, out_channels: int, bias: bool, **kwargs
    ) -> "LayerInfo":
        """
        creates a LayerInfo object for a fully connected linear layer

        :param name: unique name of the layer within its model
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param bias: True if the linear layer has a bias add included, False otherwise
        :param kwargs: additional kwargs to be passed to the LayerInfo constructor
        :return:
        """
        attributes = {
            "in_channels": in_channels,
            "out_channels": out_channels,
        }
        attributes.update(kwargs.get("attributes", {}))
        kwargs["attributes"] = attributes

        return cls(
            name=name,
            op_type="linear",
            params=in_channels * out_channels,
            bias_params=out_channels if bias else None,
            prunable=True,
            **kwargs,  # TODO: add FLOPS calculation
        )

    @classmethod
    def conv_layer(
        cls,
        name: str,
        in_channels: int,
        out_channels: int,
        kernel_shape: List[int],
        bias: bool,
        groups: int = 1,
        stride: Union[int, List[int]] = 1,
        padding: List[int] = None,
        **kwargs,
    ) -> "LayerInfo":
        """
        creates a LayerInfo object for a fully connected convolutional layer

        :param name: unique name of the layer within its model
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_shape: kernel shape of this layer, given as a list
        :param bias: True if the linear layer has a bias add included, False otherwise
        :param groups: number of groups that input and output channels are divided into.
            default is 1
        :param stride: stride for this convolution, can be int or tuple of ints. default
            is 1
        :param padding: padding applied to each spatial axis. defualt is [0, 0, 0, 0]
        :param kwargs: additional kwargs to be passed to the LayerInfo constructor
        :return:
        """
        attributes = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_shape": kernel_shape,
            "groups": groups,
            "stride": stride,
            "padding": padding if padding is not None else [0, 0, 0, 0],
        }
        attributes.update(kwargs.get("attributes", {}))
        kwargs["attributes"] = attributes

        return cls(
            name=name,
            op_type="conv",
            params=in_channels * out_channels * numpy.prod(kernel_shape) // groups,
            bias_params=out_channels if bias else None,
            prunable=True,
            **kwargs,  # TODO: add FLOPS calculation
        )


class Result(BaseModel):
    """
    Base class for storing the results of an analysis
    """

    value: Any = Field(
        title="value",
        default=None,
        description="initial value of the result",
    )
    attributes: Optional[Dict[str, Any]] = Field(
        title="attributes",
        default=None,
        description="dict of attributes of this result",
    )


class ModelResult(Result):
    """
    Class for storing the results of an analysis for an entire model
    """

    analysis_type: str = Field(
        title="analysis_type",
        description="name of the type of analysis that was performed",
    )
    layer_results: Dict[str, Result] = Field(
        title="layer_results",
        default_factory=dict,
        description=(
            "dict of layer results to initialize for this analysis. should map "
            "layer name to Result object"
        ),
    )


class PruningSensitivityResultTypes(Enum):
    """
    Types of pruning sensitivity results standardized by SparseML, used as
    ModelResult.analysis_type
    """

    LOSS = "pruning_sensitivity_loss"
    PERF = "pruning_sensitivity_perf"


class PruningSensitivityResult(ModelResult):
    """
    Helper class for creating and updating results of pruning sensitivity analyses

    :param analysis_type: PruningSensitivityResultTypes Enum value
        of type of analysis this is
    :param kwargs: optional args to be passed into model result constructor
    """

    def __init__(
        self,
        analysis_type: PruningSensitivityResultTypes,
        **kwargs,
    ):
        # validate result type
        analysis_type = PruningSensitivityResultTypes(analysis_type)
        super().__init__(analysis_type=analysis_type.value, **kwargs)

    def add_layer_sparsity_result(self, layer_name: str, sparsity: float, value: Any):
        """
        Adds a result of the given value for a given sparsity to the given layer name
        :param layer_name: layer param name to add result for
        :param sparsity: sparsity of the layer at which the sensitivity was measured
        :param value: sensitivity value
        """

        sparsity = str(sparsity)

        if layer_name not in self.layer_results:
            self.layer_results[layer_name] = Result(value={})

        self.layer_results[layer_name].value[sparsity] = value

    def add_model_sparsity_result(self, sparsity: float, value: Any):
        """
        Adds a model result of the given value for a given sparsity
        :param sparsity: sparsity of model at which the sensitivity was measured
        :param value: sensitivity value
        """

        sparsity = str(sparsity)

        if self.value is None:
            self.value = {}
        self.value[sparsity] = value

    def get_available_layer_sparsities(self) -> List[float]:
        """
        :return: list of sparsity values available for all model layers
        """
        available_sparsities = None

        for result in self.layer_results.values():
            sparsities = set(result.value.keys())
            if available_sparsities is None:
                available_sparsities = sparsities
            else:
                available_sparsities = available_sparsities.intersection(sparsities)
        return [float(sparsity) for sparsity in sorted(available_sparsities)]

    def get_layer_sparsity_score(self, layer_name: str, sparsity: float) -> float:
        """
        :param layer_name: name of layer to get sparsity score for
        :param sparsity: sparsity to measure score at
        :return: sparsity score at the given sparsity such that higher scores correlate
            to a less prunable layer
        """
        result = self.layer_results[layer_name].value

        sparsity = str(sparsity)
        if sparsity not in result:
            raise ValueError(f"No result for sparsity {sparsity} in layer {layer_name}")

        baseline_sparsity = str(0.0 if 0.0 in result else min(result))

        return (
            result[sparsity]
            if self.analysis_type is PruningSensitivityResultTypes.PERF
            else result[sparsity] - result[baseline_sparsity]
        )


_ANALYSIS_TYPE_TO_CLASS = {
    PruningSensitivityResultTypes.LOSS.value: PruningSensitivityResult,
    PruningSensitivityResultTypes.PERF.value: PruningSensitivityResult,
}


def _model_result_from_dict(model_result_dict: Dict[str, Any]) -> ModelResult:
    if "analysis_type" not in model_result_dict:
        raise ValueError(
            "'analysis_type' must be a dict key of a ModelResult dict found keys: "
            f"{list(model_result_dict.keys())}"
        )
    result_class = _ANALYSIS_TYPE_TO_CLASS.get(
        model_result_dict["analysis_type"], ModelResult
    )
    return result_class.parse_obj(model_result_dict)


class ModelInfo(ABC):
    """
    Base class for extracting and serializing model metadata, layers info, and
    analysis results

    :param model: framework specific model object to extract info for
    :param metadata: optional dict of string metadata attributes to value. Default
        is empty dict
    """

    def __init__(self, model: Any, metadata: Optional[Dict[str, Any]] = None):
        self.metadata = metadata or {}

        if _is_layer_info_dict(model):
            self._layer_info = model
        else:
            model = self.validate_model(model)
            self._layer_info = self.extract_layer_info(model)

        self._analysis_results = []  # type: List[ModelResult]

    @classmethod
    def from_dict(cls, dictionary: Dict[str, Any]):
        """
        :param dictionary: dict serialized by `dict(ModelInfo(...))`
        :return: ModelInfo object created from the given dict
        """
        dictionary = deepcopy(dictionary)
        if "layer_info" not in dictionary:
            raise ValueError(
                "ModelInfo objects serialized as a dict must include a 'layer_info' key"
            )
        layer_info = {
            name: LayerInfo.parse_obj(info)
            for name, info in dictionary["layer_info"].items()
        }

        model_info = cls(layer_info, metadata=dictionary.get("metadata", {}))

        results = dictionary.get("analysis_results", [])
        for result in results:
            model_result = _model_result_from_dict(result)
            model_info.add_analysis_result(model_result)

        return model_info

    @staticmethod
    def load(file_path) -> "ModelInfo":
        """
        :param file_path: file path to JSON file to load ModelInfo object from
        :return: the loaded ModelInfo object
        """
        file_path = clean_path(file_path)
        with open(file_path, "r") as file:
            model_info_dict = json.load(file)
        return ModelInfo.from_dict(model_info_dict)

    @property
    def layer_info(self) -> "OrderedDict[str, LayerInfo]":
        """
        :return: dict of unique layer name to LayerInfo object of the given layer
        """
        return self._layer_info

    @property
    def analysis_results(self) -> List[ModelResult]:
        """
        :return: list of analysis results run on this model
        """
        return self._analysis_results

    @abstractmethod
    def extract_layer_info(self, model: Any) -> "OrderedDict[str, LayerInfo]":
        """
        Abstract method for extracting an ordered dictionary of layer name to
        completed LayerInfo object for the layer

        :param model: model to extract LayerInfo information of
        :return: ordered dictionary of layer name to LayerInfo object for the layer
        """
        raise NotImplementedError()

    def add_analysis_result(self, result: ModelResult):
        for layer_name in result.layer_results:
            assert layer_name in self._layer_info
        self._analysis_results.append(result)

    def get_results_by_type(self, analysis_type: str) -> List[ModelResult]:
        """
        :param analysis_type: type of analysis in ModelResult.analysis_type to
            filter by
        :return: list of analysis results of this model that match the given type
        """
        return [
            result
            for result in self._analysis_results
            if result.analysis_type == analysis_type
        ]

    def get_prunable_param_names(self) -> Set[str]:
        """
        :return: set of parameter names of all prunable layers in this ModelInfo
        """
        return {
            layer_name
            for layer_name, layer_info in self.layer_info.items()
            if layer_info.prunable
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        :return: dict representation of this ModelResult
        """
        layer_info = {name: dict(info) for name, info in self._layer_info.items()}
        analysis_results = [dict(result) for result in self._analysis_results]
        return {
            "metadata": self.metadata,
            "layer_info": layer_info,
            "analysis_results": analysis_results,
        }

    def save(self, file_path: str):
        """
        saves the dict representation of this ModelInfo object as a JSON file
        to the given file path
        :param file_path: file path to save to
        """
        create_parent_dirs(file_path)
        with open(file_path, "w") as file:
            json.dump(self.to_dict(), file)

    @staticmethod
    def validate_model(model: Any) -> Any:
        # perform any validation, unwrapping, pre-processing of model
        return model


def _is_layer_info_dict(obj: Any) -> bool:
    return isinstance(obj, Dict) and all(
        isinstance(val, LayerInfo) for val in obj.values()
    )
