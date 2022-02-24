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
NMTrainingArguments class extends the original HF TrainingArguments class
to include additional arguments
"""

from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class NMTrainingArguments(TrainingArguments):
    # Add support to total batch size arguments
    total_train_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Total batch size for training."
                "If defined, is used to compute batch size per "
                "GPU/TPU core/CPU."
            )
        },
    )
    total_eval_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Total batch size for evaluation."
                "If defined, is used to compute batch size "
                "per GPU/TPU core/CPU."
            )
        },
    )

    # Override per device batch size arguments to remove default values
    per_device_train_batch_size: Optional[int] = field(
        default=None, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."},
    )

    #
    def __post_init__(self):
        # Execute post_init from TrainingArguments class
        super().__post_init__()

        # Raise error if both total and per device batch sizes are defined
        if self.total_train_batch_size and self.per_device_train_batch_size:
            raise ValueError(
                (
                    "Cannot specify both total_train_batch_size and"
                    " per_device_train_batch_size."
                )
            )
        if self.total_eval_batch_size and self.per_device_eval_batch_size:
            raise ValueError(
                (
                    "Cannot specify both total_eval_batch_size and"
                    " per_device_eval_batch_size."
                )
            )

        # If neither total nor per device batch sizes are defined
        # set total batch sizes as default
        if not self.total_train_batch_size and not self.per_device_train_batch_size:
            self.total_train_batch_size = 8
        if not self.total_eval_batch_size and not self.per_device_eval_batch_size:
            self.total_eval_batch_size = 8

        # If total batch size is defined at this point,
        # per device batch size needs to be computed
        if self.total_train_batch_size:
            self.per_device_train_batch_size = max(
                1,
                self.total_train_batch_size
                // (
                    max(1, self.n_gpu)
                    * self.gradient_accumulation_steps
                    * self.world_size
                ),
            )
        if self.total_eval_batch_size:
            self.per_device_eval_batch_size = max(
                1, self.total_eval_batch_size // (max(1, self.n_gpu) * self.world_size)
            )
