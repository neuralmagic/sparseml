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
SparseML transformers trainer class to be plugged in with existing HF trainer flows
"""


import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.file_utils import WEIGHTS_NAME

from sparseml.pytorch.optim import LayerPruningModifier, QuantizationModifier
from sparseml.pytorch.optim.manager import ScheduledModifierManager
from sparseml.pytorch.optim.optimizer import ScheduledOptimizer
from sparseml.pytorch.utils import logger
from sparseml.transformers.utils.helpers import RECIPE_NAME


__all__ = [
    "SparseMLTrainer",
    "DisableHalfPrecisionCallback",
]


_LOGGER = logging.getLogger(__name__)


class SparseMLTrainer:
    """
    Trainer for running sparsification recipes with transformers Trainer flows.

    Should either be used in place of standard transformers Trainer class
    or instantiated with multi-inheretance with a custom trainer class. SparesMLTrainer
    must be provided before Trainer for proper class dependency resolution

    i.e. class MyCustomTrainer(SparseMLTrainer, Trainer)

    :param model_name_or_path: path to model directory to be trained
    :param recipe: path to recipe for model sparsification
    :param checkpoint_recipes: list of paths to recipes used to train the
        starting checkpoint for this training run. Will be applied to the model
        on call to `apply_recipes` so that model state can be reproduced for
        weight loading
    :param teacher: teacher model for distillation. Default is None
    :param recipe_args: Dictionary of recipe variables to override or json
        loadable string of those args. Default is None
    :param teacher_input_keys: keywords of inputs to select from student inputs dict
        to also be passed to a the teacher model. Can be useful to avoid extra
        computation in forward pass that is not necessary for distillation. Defaults
        to passing all student inputs to teacher
    :param args: arguments passed into parent class
    :param kwargs: key word arguments passed to the parent class
    """

    def __init__(
        self,
        model_name_or_path: str,
        recipe: str,
        checkpoint_recipes: Union[str, List[str]] = None,
        teacher: Optional[torch.nn.Module] = None,
        recipe_args: Union[Dict[str, Any], str] = None,
        teacher_input_keys: Optional[List[str]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name_or_path = str(model_name_or_path)
        self.recipe = recipe
        self.checkpoint_recipes = list(
            [checkpoint_recipes]
            if isinstance(checkpoint_recipes, str)
            else checkpoint_recipes or []
        )  # List[str]
        self.teacher = teacher
        if self.teacher is not None:
            self.teacher.eval()
        self.teacher_input_keys = teacher_input_keys
        self.criterion = torch.nn.CrossEntropyLoss()

        if recipe_args is not None:
            if isinstance(recipe_args, str):
                recipe_args = json.loads(recipe_args)
            if not isinstance(recipe_args, Dict):
                raise ValueError("Cannot convert recipe arguments into dictionary")
        else:
            recipe_args = {}

        # initialize manager and override num epochs if available
        self.manager = ScheduledModifierManager.from_yaml(recipe, **recipe_args)
        if (
            self.manager.max_epochs
            and "args" in kwargs
            and (hasattr(kwargs["args"], "num_train_epochs"))
        ):
            kwargs["args"].num_train_epochs = self.manager.max_epochs

        self.loggers = None
        if self.recipes is not None:
            loggers = []
            if "wandb" in self.args.report_to:
                loggers.append(logger.WANDBLogger())
            self.loggers = loggers

        # add disable FP16 callback
        self.callback_handler.add_callback(DisableHalfPrecisionCallback(self))

    def apply_recipes(self, epoch=0.0):
        """
        Applies all recipes from checkpoint_recipes. Runs architecture changing
        modifiers to prepare model for state dict loading
        """
        for checkpoint_recipe in self.checkpoint_recipes:
            ScheduledModifierManager.from_yaml(checkpoint_recipe).apply(self.model)
        if self.manager is not None:
            org_state_dict = self.model.state_dict()
            self.manager.initialize(
                self.model,
                epoch=epoch,
                distillation_teacher=self.teacher,
                loggers=self.loggers,
            )
            new_state_dict = self.model.state_dict()
            new_params = [p for p in new_state_dict.keys() if p not in org_state_dict]

            if os.path.isdir(self.model_name_or_path):
                if os.path.isfile(os.path.join(self.model_name_or_path, WEIGHTS_NAME)):
                    archive_file = os.path.join(self.model_name_or_path, WEIGHTS_NAME)
                    state_dict = torch.load(archive_file, map_location="cpu")
                    new_params_to_init = [
                        p for p in new_params if p in state_dict.keys()
                    ]
                    if new_params_to_init:
                        # parameters from dict are dependent on recipe
                        (
                            _,
                            missing_keys,
                            unexpected_keys,
                            _,
                        ) = self.model._load_state_dict_into_model(
                            self.model,
                            state_dict,
                            self.model_name_or_path,
                            _fast_init=False,
                        )
                        if missing_keys or unexpected_keys:
                            raise RuntimeError(
                                "Unexpected or missing keys detected when applying "
                                f"recipes to models\nMissing keys: {missing_keys}\n"
                                f"Unexpected keys: {unexpected_keys}\n"
                            )

    def create_optimizer(self):
        """
        Create optimizer customized using SparseML
        """
        super().create_optimizer()
        if not self.recipes:
            return
        steps_per_epoch = math.ceil(
            len(self.train_dataset)
            / (self.args.per_device_train_batch_size * self.args._n_gpu)
        )
        self.args.num_train_epochs = float(self.manager.max_epochs)
        if hasattr(self, "scaler"):
            self.scaler = self.manager.modify(
                self.model,
                self.optimizer,
                steps_per_epoch=steps_per_epoch,
                wrap_optim=self.scaler,
            )
        else:
            self.optimizer = ScheduledOptimizer(
                self.optimizer,
                self.model,
                self.manager,
                steps_per_epoch=steps_per_epoch,
                loggers=self.loggers,
            )

    def create_scheduler(self, num_training_steps: int):
        """
        Override LR scheduler if the SparseML manager has LR modifiers, otherwise
        set default scheduler
        """
        if self.lr_scheduler is not None:
            # scheduler already set
            return

        if self.manager is not None and self.manager.learning_rate_modifiers:
            # allow SparseML to manage LR and set a dummy scheduler
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda _: 1.0, -1
            )
        else:
            # default scheduler
            super().create_scheduler(num_training_steps)

    def qat_active(self, epoch: int):
        if self.manager is None or not self.manager.quantization_modifiers:
            return False

        qat_start = min(
            [mod.start_epoch for mod in self.manager.quantization_modifiers]
        )

        return qat_start < epoch + 1

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computing loss using teacher/student distillation
        """
        if not self.recipes or self.teacher is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

        student_outputs = model(**inputs)
        loss = student_outputs["loss"]

        teacher_inputs = (
            inputs
            if not self.teacher_input_keys
            else {k: inputs[k] for k in self.teacher_input_keys}
        )

        steps_in_epoch = -1  # Unused
        loss = self.manager.loss_update(
            loss,
            model,
            self.optimizer,
            self.state.epoch,
            steps_in_epoch,
            global_step=self.state.global_step,
            student_outputs=student_outputs,
            teacher_inputs=teacher_inputs,
        )
        return (loss, student_outputs) if return_outputs else loss

    def save_model(self, output_dir: Optional[str] = None):
        """
        Save model during or after training. Modifiers that change the model
        architecture will also be saved
        """
        super().save_model(output_dir=output_dir)
        if self.manager is not None:
            self._save_arch_modifiers(output_dir=output_dir)

    def _save_arch_modifiers(self, output_dir: Optional[str] = None):
        """
        Save modifiers that change the model's architecture, which is to be applied
        later on whenever the model is loaded
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        output_recipe_file = os.path.join(output_dir, RECIPE_NAME)
        saved_mods = [
            mod
            for mod in self.manager.modifiers
            if isinstance(mod, QuantizationModifier)
            or isinstance(mod, LayerPruningModifier)
        ]
        if saved_mods:
            with open(output_recipe_file, "a") as yaml_file:
                for mod in saved_mods:
                    yaml_file.write(str(mod) + "\n\n")


class DisableHalfPrecisionCallback(TrainerCallback):
    """
    TrainerCallback for disabling FP16 training when QAT training begins

    :param sparseml_trainer: SparseML trainer that will call back into this object
    :param args: args to be passed to base TrainerCallback
    :param kwargs: key word arguments to be passed to base TrainerCallback
    """

    def __init__(self, sparseml_trainer: SparseMLTrainer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trainer = sparseml_trainer

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of an epoch. Disables
        """
        super().on_epoch_begin(args, state, control, **kwargs)
        if (
            hasattr(self._trainer, "scaler")
            and self._trainer.scaler._enabled
            and (self._trainer.qat_active(state.epoch))
        ):
            _LOGGER.info("entering QAT phase, disabling FP16 training")
            self.scaler._enabled = False
