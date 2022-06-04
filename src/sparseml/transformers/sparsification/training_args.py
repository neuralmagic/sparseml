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

from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments as HFTrainingArgs


__all__ = ["TrainingArguments"]


@dataclass
class TrainingArguments(HFTrainingArgs):
    """
    Training arguments specific to SparseML Transformers workflow

    :param best_model_after_epoch (`int`, *optional*, defaults to None):
        The epoch after which best model will be saved; used in conjunction
        with `load_best_model_at_end` and `metric_for_best_model` training
        arguments
    """

    distill_teacher: Optional[str] = field(
        default=None,
        metadata={
            "help": "Teacher model (a trained text classification model)",
        },
    )
    best_model_after_epoch: int = field(
        default=None,
        metadata={"help": "Epoch after which best model will be saved."},
    )
    recipe: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to a SparseML sparsification recipe, see "
                "https://github.com/neuralmagic/sparseml for more information"
            ),
        },
    )
    recipe_args: Optional[str] = field(
        default=None,
        metadata={"help": "Recipe arguments to be overwritten"},
    )
