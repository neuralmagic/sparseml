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
import shutil
import subprocess
import sys
import tempfile
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler

from sparseml.pytorch.optim.manager import ScheduledModifierManager
from sparseml.pytorch.utils.helpers import download_framework_model_by_recipe_type
from sparseml.pytorch.utils.logger import LoggerManager, PythonLogger, WANDBLogger
from sparsezoo import Model
from ultralytics import __version__
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import LOGGER
from ultralytics.yolo.utils.autobatch import check_train_batch_size
from ultralytics.yolo.utils.dist import (
    USER_CONFIG_DIR,
    ddp_cleanup,
    find_free_network_port,
)
from ultralytics.yolo.utils.torch_utils import ModelEMA, de_parallel, one_cycle
from ultralytics.yolo.v8.classify.train import ClassificationTrainer
from ultralytics.yolo.v8.detect.train import DetectionTrainer
from ultralytics.yolo.v8.segment.train import SegmentationTrainer


class _NullLRScheduler:
    def step(self):
        pass


DEFAULT_SPARSEML_CONFIG = Path(__file__).resolve().parent / "default.yaml"


class SparseTrainer(BaseTrainer):
    """
    Adds SparseML support to yolov8 BaseTrainer. This works in the following way:

    1. Override the DDP command generation that YOLO has built in,
        which assumes the trainer class is under `ultralytics` package.
    2. Override `setup_model()` to support zoo checkpoints
    3. Override `_setup_train()` to:
        1. Initialize sparseml managers/loggers
        2. Hook into checkpoint logic
        3. Override lr scheduler logic
    4. Add callbacks to properly deactivation of EMA & AMP
    5. Override `save_model()` to add manager to checkpoints
    """

    def __init__(self, config=DEFAULT_SPARSEML_CONFIG, overrides=None):
        super().__init__(config, overrides)

        if isinstance(self.model, str) and self.model.startswith("zoo:"):
            self.model = download_framework_model_by_recipe_type(Model(self.model))

        if (
            self.args.checkpoint_path is not None
            and self.args.checkpoint_path.startswith("zoo:")
        ):
            self.args.checkpoint_path = download_framework_model_by_recipe_type(
                Model(self.args.checkpoint_path)
            )

        self.manager: Optional[ScheduledModifierManager] = None
        self.checkpoint_manager: Optional[ScheduledModifierManager] = None
        self.logger_manager: LoggerManager = LoggerManager(log_python=False)

        self.epoch_step: int = 0
        self.steps_per_epoch: int = 0
        self.do_emulated_step: bool = False

        self.add_callback(
            "on_train_epoch_start", SparseTrainer.callback_on_train_epoch_start
        )
        self.add_callback(
            "on_train_batch_start", SparseTrainer.callback_on_train_batch_start
        )
        self.add_callback(
            "on_train_batch_end", SparseTrainer.callback_on_train_batch_end
        )
        self.add_callback("teardown", SparseTrainer.callback_teardown)

    def train(self):
        # NOTE: overriden to use our version of generate_ddp_command
        world_size = torch.cuda.device_count()
        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            command = generate_ddp_command(world_size, self)
            try:
                subprocess.run(command)
            except Exception as e:
                self.console(e)
            finally:
                ddp_cleanup(command, self)
        else:
            self._do_train(int(os.getenv("RANK", -1)), world_size)

    def _setup_train(self, rank, world_size):
        # NOTE: copied from BaseTrainer._setup_train with the differences:
        # 1. overrides creation of lr scheduler based on manager
        # 2. initializes sparseml with `self._initializer_sparseml`
        self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[rank])

        # Batch size
        if self.batch_size == -1:
            if rank == -1:  # single-GPU only, estimate best batch size
                self.batch_size = check_train_batch_size(
                    self.model, self.args.imgsz, self.amp
                )
            else:
                raise SyntaxError(
                    "batch=-1 to use AutoBatch is only available in "
                    "Single-GPU training. Please pass a valid batch "
                    "size value for Multi-GPU DDP training, i.e. batch=16"
                )

        # Optimizer
        self.accumulate = max(
            round(self.args.nbs / self.batch_size), 1
        )  # accumulate loss before optimizing
        self.args.weight_decay *= (
            self.batch_size * self.accumulate / self.args.nbs
        )  # scale weight_decay
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=self.args.weight_decay,
        )

        # dataloaders
        batch_size = (
            self.batch_size // world_size if world_size > 1 else self.batch_size
        )
        self.train_loader = self.get_dataloader(
            self.trainset, batch_size=batch_size, rank=rank, mode="train"
        )
        if rank in {0, -1}:
            self.test_loader = self.get_dataloader(
                self.testset, batch_size=batch_size * 2, rank=-1, mode="val"
            )
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(
                prefix="val"
            )
            self.metrics = dict(
                zip(metric_keys, [0] * len(metric_keys))
            )  # TODO: init metrics for plot_results()?
            self.ema = ModelEMA(self.model)
        self.resume_training(ckpt)
        self.run_callbacks("on_pretrain_routine_end")

        # Modification #1
        self._initializer_sparseml(rank, ckpt)

        # NOTE: we always need to populate self.lf since it is used with warmup epochs
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)
        else:
            self.lf = (
                lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf
            )

        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move

        # Modification #2 - move scheduler creation to after the manager has been
        # created and override with manager
        if self.manager is not None and self.manager.learning_rate_modifiers:
            self.scheduler = _NullLRScheduler()

    def _initializer_sparseml(self, rank, ckpt):
        if ckpt is not None:
            # resume - set manager from checkpoint
            if "recipe" not in ckpt:
                raise ValueError("resume is set not checkpoint does not have recipe")
            self.manager = ScheduledModifierManager.from_yaml(ckpt["recipe"])
        elif self.args.checkpoint_path is not None:
            # previous checkpoint
            if self.args.recipe is not None:
                self.manager = ScheduledModifierManager.from_yaml(
                    self.args.recipe, recipe_variables=self.args.recipe_args
                )
            old_ckpt = torch.load(self.args.checkpoint_path)
            if "recipe" in old_ckpt and old_ckpt["recipe"] is not None:
                self.checkpoint_manager = ScheduledModifierManager.from_yaml(
                    old_ckpt["recipe"]
                )
        elif self.args.recipe is not None:
            # normal training
            self.manager = ScheduledModifierManager.from_yaml(
                self.args.recipe, recipe_variables=self.args.recipe_args
            )

        if self.manager is not None:
            self.args.epochs = self.manager.max_epochs

        if rank in {0, -1}:
            config = dict(self.args)
            if self.manager is not None:
                config["manager"] = str(self.manager)
            loggers = [PythonLogger(logger=LOGGER)]
            try:
                loggers.append(WANDBLogger(init_kwargs=dict(config=config)))
            except ImportError:
                warnings.warn("Unable to import wandb for logging")
            self.logger_manager = LoggerManager(loggers)

        if self.manager is not None:
            self.manager.initialize(
                self.model, epoch=self.start_epoch, loggers=self.logger_manager
            )

            # NOTE: we intentionally don't divide number of batches by gradient
            # accumulation.
            # This is because yolov8 changes size of gradient accumulation during
            # warmup epochs, which is incompatible with SparseML managers
            # because they assume a static steps_per_epoch.
            # Instead, the manager will effectively ignore gradient accumulation,
            # and we will call self.scaler.emulated_step() if the batch was
            # accumulated.
            self.steps_per_epoch = len(self.train_loader)  # / self.accumulate

            self.scaler = self.manager.modify(
                self.model,
                self.optimizer,
                steps_per_epoch=self.steps_per_epoch,
                epoch=self.start_epoch,
                wrap_optim=self.scaler,
            )

    def callback_on_train_epoch_start(self):
        # NOTE: this callback is registered in __init__
        if self.manager is not None and self.manager.qat_active(epoch=self.epoch):
            if self.scaler is not None:
                self.scaler._enabled = False
            self.ema = None

        self.epoch_step = 0

    def callback_on_train_batch_start(self):
        self.do_emulated_step = True

    def optimizer_step(self):
        super().optimizer_step()
        self.do_emulated_step = False

    def callback_on_train_batch_end(self):
        if self.do_emulated_step:
            self.scaler.emulated_step()

        step = self.epoch * self.steps_per_epoch + self.epoch_step
        for key, value in self.label_loss_items(self.tloss).items():
            self.logger_manager.log_scalar(key, value, step=step)

    def save_model(self):
        epoch = -1 if self.epoch == self.epochs - 1 else self.epoch

        # NOTE: identical to super().save_model() with the addition of recipe key
        if self.checkpoint_manager is not None:
            if epoch >= 0:
                epoch += self.checkpoint_manager.max_epochs
            manager = ScheduledModifierManager.compose_staged(
                self.checkpoint_manager, self.manager
            )
        else:
            manager = self.manager if self.manager is not None else None

        ckpt = {
            "epoch": epoch,
            "best_fitness": self.best_fitness,
            "model": deepcopy(de_parallel(self.model)).half(),
            "ema": deepcopy(self.ema.ema).half(),
            "updates": self.ema.updates,
            "optimizer": self.optimizer.state_dict(),
            "train_args": self.args,
            "date": datetime.now().isoformat(),
            "version": __version__,
        }

        if manager is not None:
            ckpt["recipe"] = str(manager)

        # Save last, best and delete
        torch.save(ckpt, self.last)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
        del ckpt

    def callback_teardown(self):
        # NOTE: this callback is registered in __init__
        if self.manager is not None:
            self.manager.finalize()


class SparseDetectionTrainer(SparseTrainer, DetectionTrainer):
    pass


class SparseClassificationTrainer(SparseTrainer, ClassificationTrainer):
    pass


class SparseSegmentationTrainer(SparseTrainer, SegmentationTrainer):
    pass


def generate_ddp_command(world_size, trainer):
    # NOTE: copied from ultralytics.yolo.utils.dist.generate_ddp_command
    import __main__  # noqa local import to avoid https://github.com/Lightning-AI/lightning/issues/15218

    file_name = os.path.abspath(sys.argv[0])
    using_cli = not file_name.endswith(".py")
    if using_cli:
        file_name = generate_ddp_file(trainer)
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        f"{world_size}",
        "--master_port",
        f"{find_free_network_port()}",
        file_name,
    ] + sys.argv[1:]


def generate_ddp_file(trainer):
    # NOTE: adapted from ultralytics.yolo.utils.dist.generate_ddp_file

    if not trainer.resume:
        shutil.rmtree(trainer.save_dir)  # remove the save_dir

    content = f"""if __name__ == "__main__":
    from sparseml.pytorch.yolov8.trainers import {trainer.__class__.__name__}
    trainer = {trainer.__class__.__name__}(config={dict(trainer.args)})
    trainer.train()
"""
    (USER_CONFIG_DIR / "DDP").mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="_temp_",
        suffix=f"{id(trainer)}.py",
        mode="w+",
        encoding="utf-8",
        dir=USER_CONFIG_DIR / "DDP",
        delete=False,
    ) as file:
        file.write(content)
    return file.name
