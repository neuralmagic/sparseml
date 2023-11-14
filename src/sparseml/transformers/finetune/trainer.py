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
from typing import Any, Dict, Optional, Union

import torch
from torch.nn import Module
from transformers import Trainer as HFTransformersTrainer
from transformers.trainer_pt_utils import reissue_pt_warnings

from sparseml.transformers.finetune.session_mixin import SessionManagerMixIn


__all__ = ["Trainer"]

TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class Trainer(SessionManagerMixIn, HFTransformersTrainer):
    """
    Training implementation for running sparsification recipes with transformers flows.
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
        model: Module,
        model_state_path: str,
        recipe: Optional[str],
        recipe_args: Optional[Union[Dict[str, Any], str]] = None,
        teacher: Optional[Union[Module, str]] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            model_state_path=model_state_path,
            recipe=recipe,
            recipe_args=recipe_args,
            teacher=teacher,
            **kwargs,
        )

    def save_optimizer_and_scheduler(self, output_dir: Optional[str] = None):
        """
        Save optimizer, scheduler and scaler

        :param output_dir: The output model directory to save the above
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        # if self.sharded_ddp == ShardedDDPOption.SIMPLE and self.optimizer is not None:
        #    self.optimizer.consolidate_state_dict()

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

    def _save_checkpoint(self, model, trial, metrics=None):
        # Call into the save checkpoint by HF Transformers, which saves the
        # best metric if required
        super()._save_checkpoint(model, trial, metrics=metrics)
        if (
            self.args.metric_for_best_model is None
            or self.args.best_model_after_epoch is None
        ):
            return

        if self.state.epoch <= self.args.best_model_after_epoch:
            self.state.best_metric = None
            self.state.best_model_checkpoint = None

    def _load_optimizer_and_scheduler(self, checkpoint):
        """
        Override the Transformers Trainer so that optimizer, scheduler and scaler could
        be loaded also from the input model folder, which is our use case (instead of
        only from a separate checkpoint folder).
        """
        # We include the model path as where the optimizer and scheduler could be loaded
        # (in addition to checkpoint folders)
        model_folder = checkpoint if checkpoint is not None else self.model_state_path
        if not os.path.isfile(os.path.join(model_folder, OPTIMIZER_NAME)):
            return

        super()._load_optimizer_and_scheduler(model_folder)

        # TODO: not yet implemented
        # if self.manager.learning_rate_modifiers:
        # If LR modifiers are present in the recipe, SparseML willl take
        # control of the learning rate schedule. Therefore, set the built-in
        # scheduler to a dummy
        # self.lr_scheduler = self._dummy_lr_scheduler()

    def _dummy_lr_scheduler(self):
        return torch.optim.lr_scheduler.MultiplicativeLR(
            self.optimizer,
            lambda _: 1.0,
        )