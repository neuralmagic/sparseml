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
from ultralytics.yolo.utils import LOGGER, yaml_load
from ultralytics.yolo.utils.checks import check_yaml
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

    def update(self, model, **kwargs):
        if not self.enabled:
            return
        return super().update(model, **kwargs)

    def update_attr(self, model, **kwargs):
        if not self.enabled:
            return
        return super().update_attr(model, **kwargs)


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
            LOGGER.info("Received torch.nn.Module, not loading from checkpoint")
            self._build_managers(ckpt=None)
            return

        if not str(self.model).endswith(".pt"):
            # not a checkpoint - use ultralytics loading logic
            self.model = self.get_model(cfg=self.model, weights=None)
            self._build_managers(ckpt=None)
            return

        if not os.path.exists(str(self.model)):
            # remote ultralytics checkpoint - zoo checkpoint was already downloaded
            # in constructor
            LOGGER.info("Loading remote ultralytics checkpoint")
            weights, ckpt = attempt_load_one_weight(self.model)
            cfg = ckpt["model"].yaml
            self.model = self.get_model(cfg=cfg, weights=weights)
            self._build_managers(ckpt=ckpt)
            return ckpt

        ckpt = torch.load(str(self.model), map_location="cpu")

        if not ("source" in ckpt and ckpt["source"] == "sparseml"):
            # local ultralyltics checkpoint
            LOGGER.info("Loading local ultralytics checkpoint")
            weights, ckpt = attempt_load_one_weight(self.model)
            cfg = ckpt["model"].yaml
            self.model = self.get_model(cfg=cfg, weights=weights)
            self._build_managers(ckpt=ckpt)
            return ckpt

        # sanity check - this is one of our checkpoints
        LOGGER.info("Loading local sparseml checkpoint")
        assert ckpt["source"] == "sparseml"

        self.model = self.get_model(cfg=ckpt["model_yaml"], weights=None)

        # NOTE: this will apply structure, we need to do this before loading state dict
        self._build_managers(ckpt=ckpt)

        self.model.load_state_dict(ckpt["model"])
        LOGGER.info("Loaded previous weights from sparseml checkpoint")
        return ckpt

    def _build_managers(self, ckpt: Optional[dict]):
        if self.args.recipe is not None:
            self.manager = ScheduledModifierManager.from_yaml(
                self.args.recipe, recipe_variables=self.args.recipe_args
            )

        if ckpt is None:
            return

        if "recipe" not in ckpt:
            return

        if ckpt["epoch"] == -1:
            LOGGER.info(
                "Applying structure from completed recipe in checkpoint "
                f"at epoch {ckpt['epoch']}"
            )
            self.checkpoint_manager = ScheduledModifierManager.from_yaml(ckpt["recipe"])
            self.checkpoint_manager.apply_structure(self.model, epoch=float("inf"))

        else:
            # resuming
            # yolo will populate this when the --resume flag
            assert self.args.recipe is not None
            LOGGER.info(
                "Applying structure from un-finished recipe in checkpoint "
                f"at epoch {ckpt['epoch']}"
            )
            self.manager.apply_structure(self.model, epoch=ckpt["epoch"])

    def resume_training(self, ckpt):
        # NOTE: this method is called at the end of super()._setup_train()

        # set self.ema to None so super().resume_training() doesn't load from checkpoint
        cached_ema = self.ema
        self.ema = None
        super().resume_training(ckpt)
        self.ema = cached_ema

        # handle loading ema ourselves since we changed it state dict
        # instead of pickling
        if self.ema and ckpt and ckpt.get("ema"):
            ema = ckpt.get("ema")

            if isinstance(ema, dict):
                # this is one of our checkpoints - its a state dict
                ema_state_dict = ckpt["ema"]
            else:
                # this is a yolov8 checkpoint - its a pickled model
                ema_state_dict = ckpt["ema"].float().state_dict()

            try:
                self.ema.ema.load_state_dict(ema_state_dict)
            except RuntimeError:
                LOGGER.warning("Unable to load EMA weights - disabling EMA.")
                self.ema.enabled = False
            self.ema.updates = ckpt["updates"]

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
            self.epochs = self.manager.max_epochs

            if self.manager.learning_rate_modifiers:
                self.scheduler = _NullLRScheduler()

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
                loggers=self.logger_manager,
            )
        else:
            # initialize steps_per_epoch for logging when there's no recipe
            self.steps_per_epoch = len(self.train_loader)

    def callback_on_train_epoch_start(self):
        # NOTE: this callback is registered in __init__

        model_is_quantized = False
        if self.manager is not None and self.manager.qat_active(epoch=self.epoch):
            model_is_quantized = True
        if self.checkpoint_manager is not None and self.checkpoint_manager.qat_active(
            epoch=self.epoch + self.checkpoint_manager.max_epochs
        ):
            model_is_quantized = True

        if model_is_quantized:
            if self.scaler is not None:
                self.scaler._enabled = False
            self.ema.enabled = False

        self.epoch_step = 0

    def callback_on_train_batch_start(self):
        # we only need to do this if we have a recipe
        self.do_emulated_step = self.manager is not None

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

        self.epoch_step += 1

    def save_metrics(self, metrics):
        super().save_metrics(metrics)

        step = self.epoch * self.steps_per_epoch + self.epoch_step
        for key, value in metrics.items():
            self.logger_manager.log_scalar(key, value, step=step)

    def save_model(self):
        if self.manager is None and self.checkpoint_manager is None:
            return super().save_model()

        # NOTE: identical to super().save_model() with the addition of recipe key
        # in the checkpoint

        epoch = -1 if self.epoch == self.epochs - 1 else self.epoch

        if self.checkpoint_manager is not None:
            if epoch >= 0:
                epoch += self.checkpoint_manager.max_epochs

            if self.manager is not None:
                manager = ScheduledModifierManager.compose_staged(
                    self.checkpoint_manager, self.manager
                )
            else:
                manager = self.checkpoint_manager
        else:
            manager = self.manager if self.manager is not None else None

        model = de_parallel(self.model)
        ckpt = {
            "epoch": epoch,
            "best_fitness": self.best_fitness,
            "model": deepcopy(model).state_dict(),
            "model_yaml": dict(model.yaml),
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

    def validate(self):
        # skip validation if we are using a recipe
        if self.manager is None and self.checkpoint_manager is None:
            return super().validate()
        return {}, None

    def final_eval(self):
        # skip final eval if we are using a recipe
        if self.manager is None and self.checkpoint_manager is None:
            return super().final_eval()

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
            config = dict(self.ckpt["train_args"])
            config.pop("save_dir", None)
            self.ckpt_path = weights
            self.task = config["task"]
            self.overrides = deepcopy(config)
            self._reset_ckpt_args(self.overrides)
            (
                self.ModelClass,
                self.TrainerClass,
                self.ValidatorClass,
                self.PredictorClass,
            ) = self._guess_ops_from_task(self.task)

            self.model = self.ModelClass(dict(self.ckpt["model_yaml"]))
            if "recipe" in self.ckpt:
                manager = ScheduledModifierManager.from_yaml(self.ckpt["recipe"])
                epoch = self.ckpt.get("epoch", -1)
                if epoch < 0:
                    epoch = float("inf")
                LOGGER.info(
                    "Applying structure from sparseml checkpoint "
                    f"at epoch {self.ckpt['epoch']}"
                )
                manager.apply_structure(self.model, epoch=epoch)
            else:
                LOGGER.info("No recipe from in sparseml checkpoint")
            self.model.load_state_dict(self.ckpt["model"])
            LOGGER.info("Loaded previous weights from checkpoint")
            assert self.model.yaml == self.ckpt["model_yaml"]
        else:
            return super()._load(weights)

    def train(self, **kwargs):
        # NOTE: Copied from base class and removed post-training validation
        overrides = self.overrides.copy()
        overrides.update(kwargs)
        if kwargs.get("cfg"):
            LOGGER.info(
                f"cfg file passed. Overriding default params with {kwargs['cfg']}."
            )
            overrides = yaml_load(check_yaml(kwargs["cfg"]), append_filename=True)
        overrides["task"] = self.task
        overrides["mode"] = "train"
        if not overrides.get("data"):
            raise AttributeError(
                "dataset not provided! Please define `data` "
                "in config.yaml or pass as an argument."
            )
        if overrides.get("resume"):
            overrides["resume"] = self.ckpt_path

        self.trainer = self.TrainerClass(overrides=overrides)
        if not overrides.get("resume"):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(
                weights=self.model if self.ckpt else None, cfg=self.model.yaml
            )
            self.model = self.trainer.model
        self.trainer.train()


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
