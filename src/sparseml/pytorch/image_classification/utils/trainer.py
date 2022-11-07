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
Trainers for image classification
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import torch
from packaging import version
from torch.utils.data import DataLoader

from sparseml.pytorch.optim import ScheduledModifierManager, ScheduledOptimizer
from sparseml.pytorch.utils import (
    DEFAULT_LOSS_KEY,
    CrossEntropyLossWrapper,
    ModuleDeviceContext,
    ModuleTester,
    ModuleTrainer,
    default_device,
    download_framework_model_by_recipe_type,
    is_parallel_model,
)
from sparsezoo import Model


_LOGGER = logging.getLogger(__file__)

__all__ = [
    "ImageClassificationTrainer",
]


class Trainer(ABC):
    """
    Abstract class for Trainers
    Creates a contract that all trainers must have a run_one_epoch method
    """

    @abstractmethod
    def run_one_epoch(self):
        """
        Runs one epoch of training
        """
        raise NotImplementedError


class ImageClassificationTrainer(Trainer):
    """
    Trainer for image classification

    :param model: The loaded torch model to train
    :param key: The arch key of the model
    :param recipe_path: The path to the yaml file containing the modifiers and
        schedule to apply them with; Can also provide a SparseZoo stub prefixed
        with 'zoo:'
    :param ddp: bool indicating whether to use Distributed Data Parallel
    :param device: The device to train on Defaults to torch default device
    :param use_mixed_precision: Whether to use mixed precision FP16 training
        Defaults to False
    :param val_loader: A DataLoader for validation data
    :param train_loader: A DataLoader for training data
    :param is_main_process: Whether the current process is the main process,
        while training using DDP. Defaults to True
    :param loggers: A list of loggers to use during training process
    :param loss_fn: A Callable loss function for training and validation
        losses
    :param init_lr: The initial learning rate for the optimizer.Defaults to
        1e-9
    :param optim_name: str representing the optimizer type to use.
        Defaults to `Adam`
    :param optim_kwargs: dict of additional kwargs to pass to the optimizer
    :param recipe_args: json parsable dict of recipe variable names to values
        to overwrite with
    :param max_train_steps: The maximum number of training steps to run per epoch
        to overwrite with.
    :param one_shot: bool indicating whether to apply recipe in one shot manner
    """

    def __init__(
        self,
        model: torch.nn.Module,
        key: str,
        recipe_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        checkpoint_path: Optional[str] = None,
        ddp: bool = False,
        device: str = default_device(),
        use_mixed_precision: bool = False,
        val_loader: Optional[DataLoader] = None,
        train_loader: Optional[DataLoader] = None,
        is_main_process: bool = True,
        loggers: Optional[List[Any]] = None,
        loss_fn: Callable = lambda: CrossEntropyLossWrapper(),
        init_lr=1e-9,
        optim_name="Adam",
        optim_kwargs: Optional[Dict[str, Any]] = None,
        recipe_args: Optional[str] = None,
        max_train_steps: int = -1,
        one_shot: bool = False,
        gradient_accum_steps: int = 1,
    ):
        """
        Initializes the module_trainer
        """
        self.recipe_path = recipe_path
        self.metadata = metadata
        self.checkpoint_path = checkpoint_path
        self.ddp = ddp
        self.is_main_process = is_main_process
        self.optim_kwargs = optim_kwargs or {}
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        self.model, self.key = model, key
        self.loss_fn = loss_fn()
        self.init_lr = init_lr
        self.val_loader = val_loader
        self.train_loader = train_loader
        self.loggers = loggers
        self.recipe_args = recipe_args
        self.max_train_steps = max_train_steps
        self.one_shot = one_shot
        self._gradient_accum_steps = gradient_accum_steps

        self.val_loss = loss_fn()
        _LOGGER.info(f"created loss for validation: {self.val_loss}")

        self.train_loss = loss_fn()
        _LOGGER.info(f"created loss for training: {self.train_loss}")

        self.optim_name = optim_name
        self.epoch = 0
        self._device_context = ModuleDeviceContext(
            use_mixed_precision=self.use_mixed_precision,
        )
        if self.train_loader is not None and not self.one_shot:
            (
                self.epoch,
                self.optim,
                self.manager,
            ) = self._initialize_scheduled_optimizer()
            self.module_trainer = self._initialize_module_trainer()
        else:
            self.optim = self.manager = self.module_trainer = None
            if self.one_shot:
                self._apply_one_shot()

        self.checkpoint_manager = (
            self._setup_checkpoint_manager() if self.checkpoint_path else None
        )

        if self.val_loader is not None:
            self.module_tester = self._initialize_module_tester()
        else:
            self.module_tester = None

        self.target_metric = (
            "top1acc"
            if self.module_tester
            and "top1acc" in self.module_tester.loss.available_losses
            else DEFAULT_LOSS_KEY
        )

        if self.epoch > 0:
            _LOGGER.info("adjusting ScheduledOptimizer to restore point")
            self.optim.adjust_current_step(self.epoch, 0)

    def run_one_epoch(
        self,
        mode: str = "train",
        max_steps: Optional[int] = None,
        baseline_run: bool = False,
    ) -> Any:
        """
        Runs one epoch of training or validation

        :param mode: str representing the mode to run in, one of
            ['train', 'val']
        :param max_steps: int representing the maximum number of steps to run
            in this epoch
        :param baseline_run: bool indicating whether to run the baseline run
        :returns: Results from validation or training run
        """
        train_mode = mode == "train"
        validation_mode = mode == "val"
        if not (train_mode or validation_mode):
            raise ValueError(f"Invalid train mode '{mode}', must be 'train' or 'val'")

        if (
            version.parse(torch.__version__) < version.parse("1.9")
            and self.manager
            and (self.manager.qat_active(epoch=self.epoch))
        ):
            # switch off fp16
            self._device_context.use_mixed_precision = False

        if validation_mode:
            return self._run_validation_epoch(
                max_steps=max_steps,
                baseline_run=baseline_run,
            )

        if train_mode:
            return self._run_train_epoch(max_steps=max_steps)

    @property
    def max_epochs(self):
        """
        :return: the maximum number of epochs from manager
        """
        return self.manager.max_epochs if self.manager is not None else 0

    def _apply_one_shot(self):
        self.manager = ScheduledModifierManager.from_yaml(
            self.recipe_path,
        )

        self.manager.apply(self.model)
        _LOGGER.info(f"Applied {self.recipe_path} to manager")

    def _initialize_module_tester(self):
        tester = ModuleTester(
            module=self.model,
            device=self.device,
            loss=self.val_loss,
            loggers=self.loggers,
            log_steps=-1,
        )
        return tester

    def _initialize_scheduled_optimizer(self):
        # optimizer setup
        optim_constructor = torch.optim.__dict__[self.optim_name]
        optim = optim_constructor(
            self.model.parameters(), lr=self.init_lr, **self.optim_kwargs
        )
        _LOGGER.info(f"created optimizer: {optim}")
        _LOGGER.info(
            "note, the lr for the optimizer may not reflect the manager "
            "yet until the recipe config is created and run"
        )

        epoch = 0

        manager = ScheduledModifierManager.from_yaml(
            file_path=self.recipe_path,
            recipe_variables=self.recipe_args,
            metadata=self.metadata,
        )

        steps_per_epoch = (
            len(self.train_loader) if self.max_train_steps < 0 else self.max_train_steps
        )
        optim = ScheduledOptimizer(
            optim,
            self.model.module if is_parallel_model(self.model) else self.model,
            manager,
            steps_per_epoch=steps_per_epoch,
            loggers=self.loggers,
        )
        _LOGGER.info(f"created manager: {manager}")
        return epoch, optim, manager

    def _initialize_module_trainer(self):

        trainer = ModuleTrainer(
            module=self.model,
            device=self.device,
            loss=self.train_loss,
            optimizer=self.optim,
            loggers=self.loggers,
            device_context=self._device_context,
            num_accumulated_batches=self._gradient_accum_steps,
        )
        _LOGGER.info(f"created Module Trainer: {trainer}")

        return trainer

    def _run_validation_epoch(
        self,
        max_steps: Optional[int] = None,
        baseline_run: bool = False,
    ):
        # Note: This method should not be called directly,
        # use run_one_epoch instead

        if self.is_main_process:
            assert self.module_tester, "module_tester is not initialized"

            return self.module_tester.run_epoch(
                self.val_loader,
                epoch=-1 if baseline_run else self.epoch,
                max_epochs=-1 if baseline_run else self.max_epochs,
                max_steps=max_steps,
            )

    def _run_train_epoch(
        self,
        max_steps: Optional[int] = None,
    ):
        # Note: This method should not be called directly,
        # use run_one_epoch instead

        if max_steps and max_steps > 0:
            # correct since all optimizer steps are not
            # taken in the epochs for debug mode
            self.optim.adjust_current_step(self.epoch, 0)

        if self.ddp:  # sync DDP dataloaders
            assert hasattr(self.train_loader.sampler, "set_epoch")
            self.train_loader.sampler.set_epoch(self.epoch)

        return self.module_trainer.run_epoch(
            data_loader=self.train_loader,
            epoch=self.epoch,
            max_epochs=self.max_epochs,
            max_steps=max_steps,
            show_progress=self.is_main_process,
        )

    def _setup_checkpoint_manager(self):
        if self.checkpoint_path and self.checkpoint_path.startswith("zoo:"):
            zoo_model = Model(self.checkpoint_path)
            self.checkpoint_path = download_framework_model_by_recipe_type(zoo_model)

        checkpoint_state = torch.load(self.checkpoint_path)
        checkpoint_manager = None
        checkpoint_recipe = checkpoint_state.get("recipe")
        if checkpoint_recipe:
            checkpoint_manager = ScheduledModifierManager.from_yaml(checkpoint_recipe)
        return checkpoint_manager
