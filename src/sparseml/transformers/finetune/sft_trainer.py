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

import os
import warnings
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch.nn import Module
from transformers.trainer_pt_utils import reissue_pt_warnings

from peft import PeftConfig
from sparseml.transformers.finetune.session_mixin import SessionManagerMixIn
from trl import SFTTrainer as TRLSFTTrainer


__all__ = ["SFTTrainer"]

TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class SFTTrainer(SessionManagerMixIn, TRLSFTTrainer):
    """
    Training implementation for running sparsification recipes with HF Trainer.

    :param model: the model to use with the trainer and apply sparsification to
    :param model_state_path: the state path to the model,
        used to load config and tokenizer settings
    :param recipe: the recipe, if any, to apply to the modle and training
        process
    :param recipe_args: A json string, csv key=value string, or dictionary containing
        arguments to override the root arguments within the recipe such as
        learning rate or num epochs
    :param teacher: teacher model for distillation. Set to 'self' to distill
        from the loaded model or 'disable' to turn of distillation
    :param kwargs: key word arguments passed to the parent class
    """

    def __init__(
        self,
        model_state_path: str,
        model: Optional[Module] = None,
        model_init: Optional[Callable] = None,
        recipe: Optional[str] = None,
        recipe_args: Optional[Union[Dict[str, Any], str]] = None,
        teacher: Optional[Union[Module, str]] = None,
        peft_config: Optional[PeftConfig] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            model_init=model_init,
            model_state_path=model_state_path,
            recipe=recipe,
            recipe_args=recipe_args,
            teacher=teacher,
            peft_config=peft_config,
            **kwargs,
        )

    def save_optimizer_and_scheduler(self, output_dir: Optional[str] = None):
        """
        Save optimizer, scheduler and scaler

        :param output_dir: The output model directory to save the above
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        if self.is_world_process_zero():
            if self.optimizer is not None:
                torch.save(
                    self.optimizer.state_dict(),
                    os.path.join(output_dir, "optimizer.pt"),
                )
            with warnings.catch_warnings(record=True) as caught_warnings:
                if self.lr_scheduler is not None:
                    torch.save(
                        self.lr_scheduler.state_dict(),
                        os.path.join(output_dir, "scheduler.pt"),
                    )
            reissue_pt_warnings(caught_warnings)
            if self.use_cuda_amp:
                torch.save(
                    self.scaler.state_dict(), os.path.join(output_dir, "scaler.pt")
                )

    def _prepare_dataset(
        self,
        dataset,
        tokenizer,
        packing,
        dataset_text_field,
        max_seq_length,
        formatting_func,
        num_of_sequences,
        chars_per_token,
        remove_unused_columns=True,
        append_concat_token=True,
        add_special_tokens=True,
    ):
        return dataset
