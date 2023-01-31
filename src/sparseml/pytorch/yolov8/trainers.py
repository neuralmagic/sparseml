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

from sparseml.pytorch.optim.manager import ScheduledModifierManager
from sparseml.pytorch.utils.helpers import download_framework_model_by_recipe_type
from sparseml.pytorch.utils.logger import LoggerManager, PythonLogger, WANDBLogger
from sparsezoo import Model
from ultralytics import __version__
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.yolo.engine.model import YOLO
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import LOGGER
from ultralytics.yolo.utils.dist import (
    USER_CONFIG_DIR,
    ddp_cleanup,
    find_free_network_port,
)
from ultralytics.yolo.utils.torch_utils import ModelEMA, de_parallel
from ultralytics.yolo.v8.classify.train import ClassificationTrainer
from ultralytics.yolo.v8.detect.train import DetectionTrainer
from ultralytics.yolo.v8.segment.train import SegmentationTrainer


class _NullLRScheduler:
    def step(self):
        pass


class _ToggleableEMA(ModelEMA):
    def __init__(self, ema: ModelEMA):
        self.ema = ema.ema
        self.updates = ema.updates
        self.decay = ema.decay
        self.enabled = True

    def update(self, **kwargs):
        if not self.enabled:
            return
        return super().update(**kwargs)

    def update_attr(self, **kwargs):
        if not self.enabled:
            return
        return super().update_attr(**kwargs)


DEFAULT_SPARSEML_CONFIG = Path(__file__).resolve().parent / "default.yaml"


class SparseTrainer(BaseTrainer):
    """
    Adds SparseML support to yolov8 BaseTrainer. This works in the following way:

    1. Handle zoo stubs in `__init__`
    2. Override `train()` to update the DDP command generation that YOLO has built in,
        which assumes the trainer class is under `ultralytics` package.
    3. Override `resume_training()` to create/load sparseml managers
    3. Override `_setup_train()` to:
        1. Override lr scheduler logic
        2. Initializer our sparseml managers
    4. Add callbacks to properly deactivation of EMA & AMP
    5. Override `save_model()` to add manager to checkpoints
    """

    def __init__(self, config=DEFAULT_SPARSEML_CONFIG, overrides=None):
        super().__init__(config, overrides)

        if isinstance(self.model, str) and self.model.startswith("zoo:"):
            self.model = download_framework_model_by_recipe_type(
                Model(self.model), model_suffix=".pt"
            )
            self.is_sparseml_checkpoint = True
        else:
            self.is_sparseml_checkpoint = False

        if (
            self.args.checkpoint_path is not None
            and self.args.checkpoint_path.startswith("zoo:")
        ):
            self.args.checkpoint_path = download_framework_model_by_recipe_type(
                Model(self.args.checkpoint_path), model_suffix=".pt"
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
        # NOTE: overriden to use our version of `generate_ddp_command`
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

    def setup_model(self):
        # NOTE: override to handle pickled checkpoints and our own checkpoints
        if isinstance(self.model, torch.nn.Module):
            return

        model, weights = self.model, None
        ckpt = None
        if str(model).endswith(".pt"):
            if self.is_sparseml_checkpoint or os.path.exists(str(model)):
                ckpt = torch.load(str(model), map_location="cpu")
                if "source" in ckpt and ckpt["source"] == "sparseml":
                    # this is one of our checkpoints
                    weights = ckpt["model"]
                    cfg = ckpt["model_yaml_config"]
                else:
                    # a ultralytics checkpoint
                    weights, ckpt = attempt_load_one_weight(model)
                    cfg = ckpt["model"].yaml
            else:
                weights, ckpt = attempt_load_one_weight(model)
                cfg = ckpt["model"].yaml
        else:
            cfg = model
        self.model = self.get_model(cfg=cfg, weights=weights)
        return ckpt

    def resume_training(self, ckpt):
        # NOTE: this method is called at the end of super()._setup_train()

        # set self.ema to None so super().resume_training() doesn't load from checkpoint
        cached_ema = self.ema
        self.ema = None
        super().resume_training(ckpt)
        self.ema = cached_ema

        # handle loading ema ourselves since we changed it state dict
        # instead of pickling
        if self.ema and ckpt.get("ema"):
            ema = ckpt.get("ema")

            if isinstance(ema, dict):
                # this is one of our checkpoints - its a state dict
                ema_state_dict = ckpt["ema"]
            else:
                # this is a yolov8 checkpoint - its a pickled model
                ema_state_dict = ckpt["ema"].float().state_dict()

            self.ema.ema.load_state_dict(ema_state_dict)
            self.ema.updates = ckpt["updates"]

        if ckpt is not None:
            # resume - set manager from checkpoint
            if "recipe" in ckpt:
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

    def _setup_train(self, rank, world_size):
        super()._setup_train(rank, world_size)
        # NOTE: self.resume_training() was called in ^

        if self.ema is not None:
            self.ema = _ToggleableEMA(self.ema)

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
            self.args.epochs = self.manager.max_epochs

            if self.manager.learning_rate_modifiers:
                self.scheduler = _NullLRScheduler()

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
            self.ema.enabled = False

        self.epoch_step = 0

    def callback_on_train_batch_start(self):
        self.do_emulated_step = True

    def optimizer_step(self):
        super().optimizer_step()
        self.do_emulated_step = False

    def callback_on_train_batch_end(self):
        # Here is where we handle the changing gradient accumulation values by doing
        # an emulated step if we accumulated gradients of this batch
        if self.do_emulated_step:
            self.scaler.emulated_step()

        step = self.epoch * self.steps_per_epoch + self.epoch_step
        for key, value in self.label_loss_items(self.tloss).items():
            self.logger_manager.log_scalar(key, value, step=step)

    def save_model(self):
        # NOTE: identical to super().save_model() with the addition of recipe key
        # in the checkpoint

        epoch = -1 if self.epoch == self.epochs - 1 else self.epoch

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
            "model": deepcopy(de_parallel(self.model)).state_dict(),
            "model_yaml_config": self.model.yaml,
            "ema": deepcopy(self.ema.ema).state_dict(),
            "updates": self.ema.updates,
            "optimizer": self.optimizer.state_dict(),
            "train_args": self.args,
            "date": datetime.now().isoformat(),
            "version": __version__,
            "source": "sparseml",
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


class SparseYOLO(YOLO):
    def __init__(self, model="yolov8n.yaml", type="v8") -> None:
        model_str = str(model)

        if model_str.startswith("zoo:"):
            model = download_framework_model_by_recipe_type(
                Model(model_str), model_suffix=".pt"
            )
            self.is_sparseml_checkpoint = True
        elif model_str.endswith(".pt"):
            if os.path.exists(model_str):
                ckpt = torch.load(model_str)
                self.is_sparseml_checkpoint = (
                    "source" in ckpt and ckpt["source"] == "sparseml"
                )
            else:
                self.is_sparseml_checkpoint = False
        else:
            self.is_sparseml_checkpoint = False

        super().__init__(model, type)

        if self.TrainerClass == DetectionTrainer:
            self.TrainerClass = SparseDetectionTrainer
        elif self.TrainerClass == ClassificationTrainer:
            self.TrainerClass = SparseClassificationTrainer
        elif self.TrainerClass == SegmentationTrainer:
            self.TrainerClass = SparseSegmentationTrainer

    def _load(self, weights: str):
        if self.is_sparseml_checkpoint:
            """
            NOTE: the model is given to the trainer class with this snippet
            from YOLO base class:
            ```python
            self.trainer = self.TrainerClass(overrides=overrides)
            if not overrides.get("resume"):  # manually set model only if not resuming
                self.trainer.model = self.trainer.get_model(
                    weights=self.model if self.ckpt else None,
                    cfg=self.model.yaml
                )
                self.model = self.trainer.model
            ```
            """

            self.ckpt = torch.load(weights, map_location="cpu")
            self.model = self.ckpt["model"]
            setattr(self.model, "yaml", self.ckpt["model_yaml_config"])
            self.ckpt_path = weights
            self.task = self.model["task"]
            self.overrides = deepcopy(self.model)
            self._reset_ckpt_args(self.overrides)
            (
                self.ModelClass,
                self.TrainerClass,
                self.ValidatorClass,
                self.PredictorClass,
            ) = self._guess_ops_from_task(self.task)
        else:
            return super()._load(weights)


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
