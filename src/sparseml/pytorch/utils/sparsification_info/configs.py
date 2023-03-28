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

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Dict, Generator, Tuple, Union

import torch.nn
from pydantic import BaseModel, Field

from sparseml.pytorch.utils.sparsification_info.helpers import (
    get_leaf_operations,
    get_precision_information,
    is_quantized,
)


__all__ = [
    "SparsificationSummaries",
    "SparsificationPruning",
    "SparsificationQuantization",
    "SparsificationInfo",
]


class SparsificationInfo(BaseModel, ABC):
    @classmethod
    @abstractmethod
    def from_module(
        cls,
        module: torch.nn.Module,
        **kwargs,
    ) -> "SparsificationInfo":
        """
        Factory method to create SparsificationInfo object from a module.

        :param module: The module to create the SparsificationInfo object from.
        :param kwargs: Additional arguments to pass to the SparsificationInfo object.
        :return: A SparsificationInfo object.
        """
        raise NotImplementedError()

    @abstractmethod
    def loggable_items(
        self,
    ) -> Generator[Tuple[str, Union[Dict[str, int], float, int]], None, None]:
        """
        Yield the loggable items for SparsificationInfo object.

        :return: A generator that yields the loggable items for this object.
        """
        raise NotImplementedError()


class CountAndPercent(BaseModel):
    count: int = Field(description="The count of items")
    percent: float = Field(description="The percentage of those items out of the total")


class SparsificationSummaries(SparsificationInfo):
    """
    A model that contains the sparsification summaries for a torch module.
    """

    quantized: CountAndPercent = Field(
        description="A model that contains the number of "
        "operations/the percent of operations that are quantized."
    )
    pruned: CountAndPercent = Field(
        description="A model that contains the number of "
        "parameters/the percent of parameters that are pruned."
    )
    parameter_counts: Dict[str, int] = Field(
        description="A dictionary that maps the name of a parameter "
        "to the number of elements (weights) in that parameter."
    )
    operation_counts: Dict[str, int] = Field(
        description="A dictionary that maps the name of an operation "
        "to the number of times that operation is used in the model."
    )

    @classmethod
    def from_module(
        cls,
        module=torch.nn.Module,
        pruning_thresholds: Tuple[float, float] = (0.05, 1 - 1e-9),
    ) -> "SparsificationSummaries":
        """
        Factory method to create a SparsificationSummaries object from a module.

        :param module: The module to create the SparsificationSummaries object from.
        :param pruning_thresholds: The lower and upper thresholds used to determine
            whether a parameter is pruned. If it's percentage of zero weights is between
            the lower and upper thresholds, it is considered pruned.
        :return: A SparsificationSummaries object.
        """
        operations = get_leaf_operations(module)
        num_quantized_ops = sum([is_quantized(op) for op in operations])
        total_num_params = len(list(module.parameters()))

        lower_threshold_pruning = min(pruning_thresholds)
        upper_threshold_pruning = max(pruning_thresholds)
        total_num_params_pruned = 0
        count_parameters = defaultdict(int)

        for param_name, param in module.named_parameters():
            num_parameters = param.numel()
            num_zero_parameters = param.numel() - param.count_nonzero().item()

            if (
                lower_threshold_pruning
                <= num_zero_parameters / num_parameters
                <= upper_threshold_pruning
            ):
                total_num_params_pruned += 1

            count_parameters[param_name] = num_parameters

        return cls(
            pruned=CountAndPercent(
                count=total_num_params_pruned,
                percent=total_num_params_pruned / total_num_params,
            ),
            quantized=CountAndPercent(
                count=num_quantized_ops, percent=num_quantized_ops / len(operations)
            ),
            parameter_counts=count_parameters,
            operation_counts=Counter([op.__class__.__name__ for op in operations]),
        )

    def loggable_items(
        self,
    ) -> Generator[Tuple[str, Union[Dict[str, int], float, int]], None, None]:
        """
        Yield the loggable items for SparsificationSummaries object.

        :return: A generator that yields the loggable items for this object.
        """
        main_tag = self.__class__.__name__
        yield f"{main_tag}/OperationCounts", self.operation_counts
        yield f"{main_tag}/ParameterCounts", self.parameter_counts
        yield f"{main_tag}/QuantizedOperations/count", self.quantized.count
        yield f"{main_tag}/QuantizedOperations/percent", self.quantized.percent
        yield f"{main_tag}/PrunedParameters/count", self.pruned.count
        yield f"{main_tag}/PrunedParameters/percent", self.pruned.percent


class SparsificationPruning(SparsificationInfo):
    """
    A model that contains the pruning information for a torch module.
    """

    sparse_parameters: Dict[str, CountAndPercent] = Field(
        description="A dictionary that maps the name of a parameter "
        "to the number/percent of weights that are zeroed out "
        "in that layer."
    )

    @classmethod
    def from_module(cls, module: torch.nn.Module) -> "SparsificationPruning":
        """
        Factory method to create a SparsificationPruning object from a module.

        :param module: The module to create the SparsificationPruning object from.
        :return: A SparsificationPruning object.
        """
        sparse_parameters_count = defaultdict(CountAndPercent)
        for param_name, param in module.named_parameters():
            num_parameters = param.numel()
            num_zero_parameters = param.numel() - param.count_nonzero().item()

            zero_count = num_zero_parameters
            zero_count_percent = num_zero_parameters / num_parameters

            sparse_parameters_count[param_name] = CountAndPercent(
                count=zero_count, percent=zero_count_percent
            )

        return cls(sparse_parameters=sparse_parameters_count)

    def loggable_items(
        self,
    ) -> Generator[Tuple[str, Union[Dict[str, int], float, int]], None, None]:
        """
        Yield the loggable items for SparsificationPruning object.

        :return: A generator that yields the loggable items for this object.
        """
        main_tag = self.__class__.__name__
        for param_name, count_and_percent in self.sparse_parameters.items():
            yield f"{main_tag}/SparseParameters/{param_name}/count", count_and_percent.count  # noqa: E501
            yield f"{main_tag}/SparseParameters/{param_name}/percent", count_and_percent.percent  # noqa: E501


class SparsificationQuantization(SparsificationInfo):
    """
    A model that contains the quantization information for a torch module.
    """

    enabled: Dict[str, bool] = Field(
        description="A dictionary that maps the name of an "
        "operation to a boolean flag that indicates whether "
        "the operation is quantized or not."
    )
    precision: Dict[str, Union[BaseModel, None, int]] = Field(
        description="A dictionary that maps the name of a layer"
        "to the precision of that layer."
    )

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_module(
        cls,
        module: torch.nn.Module,
    ) -> "SparsificationQuantization":
        """
        Factory method to create a SparsificationQuantization object from a module.

        :param module: The module to create the SparsificationQuantization object from.
        :return: A SparsificationQuantization object.
        """
        operations = get_leaf_operations(module)
        enabled = defaultdict(bool)
        precision = defaultdict(str)
        for op in operations:
            operation_name = op.__class__.__name__
            operation_counter = 0
            # make sure that the operation name is unique
            while enabled.get(operation_name) is not None:
                operation_counter += 1
                operation_name = f"{op.__class__.__name__}_{operation_counter}"

            enabled[operation_name] = is_quantized(op)
            precision[operation_name] = get_precision_information(op)

        return cls(enabled=enabled, precision=precision)

    def loggable_items(
        self,
    ) -> Generator[Tuple[str, Union[Dict[str, int], float, int]], None, None]:
        """
        Yield the loggable items for SparsificationQuantization object.

        :return: A generator that yields the loggable items for this object.
        """
        main_tag = self.__class__.__name__
        for operation in self.enabled.keys():
            yield f"{main_tag}/{operation}/enabled", self.enabled[operation]

            precision = self.precision[operation]
            if precision is None:
                yield f"{main_tag}/{operation}/precision", precision
            elif isinstance(precision, int):
                yield f"{main_tag}/{operation}/precision.weights/num_bits", precision
            elif isinstance(precision, BaseModel):
                yield f"{main_tag}/{operation}/precision/weights/num_bits", precision.weights.num_bits  # noqa: E501
                yield f"{main_tag}/{operation}/precision/input_activations/num_bits", precision.input_activations.num_bits  # noqa: E501
            else:
                raise ValueError(
                    f"The precision is not a valid type {type(precision)}."
                )
