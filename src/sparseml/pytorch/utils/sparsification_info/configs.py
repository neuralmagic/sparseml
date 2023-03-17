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

from typing import Any, Dict, Tuple

from pydantic import BaseModel, Field


class SparsificationSummaries(BaseModel):
    quantization: Tuple[int, float] = Field(
        description="A tuple that displays of the number of "
        "layers/the percent of layers that are quantized."
    )
    pruning: Tuple[int, float] = Field(
        description="A tuple that displays of the number of "
        "layers/the percent of layers that are pruned."
    )
    ops: Dict[str, int] = Field(
        description="A dictionary that holds the counts for "
        "each type of operation present in the module."
    )


class SparsificationPruning(BaseModel):
    zero_count_percent: Dict[str, float] = Field(
        description="A dictionary that maps the name of a layer "
        "to the percent of weights that are zeroed out "
        "in that layer."
    )
    zero_count: Dict[str, int] = Field(
        description="A dictionary that maps the name of a layer "
        "to the number of weights that are zeroed out "
        "in that layer."
    )


class SparsificationQuantization(BaseModel):
    enabled: Dict[str, bool] = Field(
        description="A dictionary that maps the name of a layer "
        "to whether or not that layer is quantized."
    )
    precision: Dict[str, Any] = Field(
        description="A dictionary that maps the name of a layer "
        "to the precision of that layer."
    )


class SparsificationDistillation(BaseModel):
    losses: Dict[str, str] = Field(
        description="A dictionary that maps the name of a layer "
        "to the loss function used for that layer."
    )
