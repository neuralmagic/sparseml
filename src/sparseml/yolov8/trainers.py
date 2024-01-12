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
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from copy import copy, deepcopy
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import List, Optional

import torch

from sparseml.optim.helpers import load_recipe_yaml_str
from sparseml.pytorch.optim.manager import ScheduledModifierManager
from sparseml.pytorch.sparsification.quantization import skip_onnx_input_quantize
from sparseml.pytorch.utils import ModuleExporter
from sparseml.pytorch.utils.helpers import download_framework_model_by_recipe_type
from sparseml.pytorch.utils.logger import LoggerManager, PythonLogger, WANDBLogger
from sparseml.yolov8.modules import Bottleneck, Conv
from sparseml.yolov8.utils import (
    check_coco128_segmentation,
    create_grad_sampler,
    data_from_dataset_path,
)
from sparseml.yolov8.utils.export_samples import export_sample_inputs_outputs
from sparseml.yolov8.validators import (
    SparseClassificationValidator,
    SparseDetectionValidator,
    SparseSegmentationValidator,
)
from sparsezoo import Model
from sparsezoo.utils import validate_onnx
from ultralytics import __version__
from ultralytics.nn.modules import Detect, Segment
from ultralytics.nn.tasks import SegmentationModel, attempt_load_one_weight
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data.dataloaders.v5loader import create_dataloader
from ultralytics.yolo.engine.model import TASK_MAP, YOLO
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import LOGGER, IterableSimpleNamespace, yaml_load
from ultralytics.yolo.utils.checks import check_file, check_imgsz, check_yaml
from ultralytics.yolo.utils.dist import (
    USER_CONFIG_DIR,
    ddp_cleanup,
    find_free_network_port,
)
from ultralytics.yolo.utils.files import get_latest_run
from ultralytics.yolo.utils.torch_utils import (
    TORCH_1_9,
    de_parallel,
    smart_inference_mode,
)
from ultralytics.yolo.v8.classify import ClassificationTrainer, ClassificationValidator
from ultralytics.yolo.v8.detect import DetectionTrainer, DetectionValidator
from ultralytics.yolo.v8.segment import SegmentationTrainer, SegmentationValidator


class _NullLRScheduler:
    def step(self):
        pass


DEFAULT_SPARSEML_CONFIG_PATH = Path(__file__).resolve().parent / "default.yaml"
DEFAULT_CFG_DICT = yaml_load(DEFAULT_SPARSEML_CONFIG_PATH)
for k, v in DEFAULT_CFG_DICT.items():
    if isinstance(v, str) and v.lower() == "none":
        DEFAULT_CFG_DICT[k] = None
DEFAULT_CFG_KEYS = DEFAULT_CFG_DICT.keys()
DEFAULT_CFG = IterableSimpleNamespace(**DEFAULT_CFG_DICT)


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

    def __init__(self, config=DEFAULT_SPARSEML_CONFIG_PATH, overrides=None):
        super().__init__(config, overrides)

        if isinstance(self.model, str) and self.model.startswith("zoo:"):
            self.model = download_framework_model_by_recipe_type(
                Model(self.model), model_suffix="pt"
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
        self.add_callback(
            "on_train_epoch_end", SparseTrainer.callback_on_train_epoch_end
        )
        self.add_callback("teardown", SparseTrainer.callback_teardown)

    def check_resume(self):
        # see note for what is different
        resume = self.args.resume
        if resume:
            try:
                last = Path(
                    check_file(resume)
                    if isinstance(resume, (str, Path)) and Path(resume).exists()
                    else get_latest_run()
                )

                # NOTE: here is the single change to this function
                # self.args = get_cfg(attempt_load_weights(last).args)
                self.args = torch.load(last)["train_args"]
                self.args = IterableSimpleNamespace(**self.args)

                self.args.model, resume = str(last), True  # reinstate
            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found. "
                    "Please pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e
        self.resume = resume

    def train(self):
        # NOTE: overriden to use our version of `generate_ddp_command`
        world_size = torch.cuda.device_count()
        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            if self.args.rect:
                LOGGER.warning(
                    "WARNING ⚠️ 'rect=True' is incompatible with Multi-GPU training, "
                    "setting rect=False"
                )
                self.args.rect = False

            command, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(f"DDP command: {command}")
                subprocess.run(command, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(command, str(file))
        else:
            self._do_train(world_size)

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
            if self.manager.quantization_modifiers:
                _modify_arch_for_quantization(self.model)

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
            if self.checkpoint_manager.quantization_modifiers:
                _modify_arch_for_quantization(self.model)
            self.checkpoint_manager.apply_structure(self.model, epoch=float("inf"))

        else:
            # resuming
            # yolo will populate this when the --resume flag
            assert self.args.recipe is not None
            LOGGER.info(
                "Applying structure from un-finished recipe in checkpoint "
                f"at epoch {ckpt['epoch']}"
            )
            if self.manager.quantization_modifiers:
                _modify_arch_for_quantization(self.model)
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

    def _setup_train(self, world_size):
        rank = int(os.getenv("RANK", -1))

        super()._setup_train(world_size)
        # NOTE: self.resume_training() was called in ^

        if rank in {0, -1}:
            self.test_loader = self.get_dataloader(
                self.testset,
                batch_size=max(1, self.train_loader.batch_size // 4),
                rank=-1,
                mode="val",
            )
            self.validator = self.get_validator()

        if rank in {0, -1}:
            config = dict(self.args)
            if self.manager is not None:
                config["manager"] = str(self.manager)
            loggers = [PythonLogger(logger=LOGGER)]
            try:
                init_kwargs = dict(config=config)
                if self.args.project is not None:
                    init_kwargs["project"] = self.args.project
                if self.args.name is not None:
                    init_kwargs["name"] = self.args.name
                loggers.append(WANDBLogger(init_kwargs=init_kwargs))
            except ImportError:
                warnings.warn("Unable to import wandb for logging")
            self.logger_manager = LoggerManager(loggers)

        if self.args.recipe is not None:
            base_path = os.path.join(self.save_dir, "original_recipe.yaml")
            with open(base_path, "w") as fp:
                fp.write(load_recipe_yaml_str(self.args.recipe))
            self.logger_manager.save(base_path)

            full_path = os.path.join(self.save_dir, "final_recipe.yaml")
            self.manager.save(full_path)
            self.logger_manager.save(full_path)

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
                self.model.module if hasattr(self.model, "module") else self.model,
                self.optimizer,
                steps_per_epoch=self.steps_per_epoch,
                epoch=self.start_epoch,
                wrap_optim=self.scaler,
                loggers=self.logger_manager,
                grad_sampler={
                    "data_loader_builder": self._get_data_loader_builder(),
                    "loss_function": lambda preds, batch: self.model.loss(
                        batch=batch, preds=preds
                    )[0]
                    / self.train_loader.batch_size,
                },
            )
        else:
            # initialize steps_per_epoch for logging when there's no recipe
            self.steps_per_epoch = len(self.train_loader)

    def _setup_ddp(self, world_size):
        # increases the timeout for DDP processes
        rank = int(os.getenv("RANK", -1))
        torch.cuda.set_device(rank)
        self.device = torch.device("cuda", rank)
        LOGGER.info(
            f"DDP settings: RANK {rank}, WORLD_SIZE {world_size}, DEVICE {self.device}"
        )
        torch.distributed.init_process_group(
            "nccl" if torch.distributed.is_nccl_available() else "gloo",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=7200),
        )

    def _get_data_loader_builder(self):
        train_loader = self.train_loader

        def _data_loader_builder(kwargs):
            template = dict(train_loader.__dict__)
            # drop attributes that will be auto-initialized
            to_drop = [
                k
                for k in template
                if k.startswith("_") or k in ["batch_sampler", "iterator", "sampler"]
            ]
            for item in to_drop:
                template.pop(item)

            # override defaults if kwargs are given, for example via recipe
            if kwargs:
                template.update(kwargs)
            data_loader = type(train_loader)(**template)

            while True:
                for batch in data_loader:
                    batch = self.preprocess_batch(batch)
                    assert batch["img"].device.index == self.device.index
                    yield [batch["img"]], {}, batch

        return _data_loader_builder

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
            if self.ema is not None:
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

    def callback_on_train_epoch_end(self):
        # NOTE: this is called right before  validation occurs
        if self.ema is not None and self.ema.enabled and self.manager is not None:
            # ema update was just called in super().optimizer_step()
            # we need to update ema's mask
            ema_state_dict = self.ema.ema.state_dict()
            for pruning_modifier in self.manager.pruning_modifiers:
                if pruning_modifier.enabled:
                    for key, mask in pruning_modifier.state_dict().items():
                        param_name = key.replace(".sparsity_mask", "")
                        ema_state_dict[param_name] *= mask

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
            "ema": deepcopy(self.ema.ema).state_dict()
            if self.ema and self.ema.enabled
            else None,
            "updates": self.ema.updates if self.ema and self.ema.enabled else None,
            "optimizer": self.optimizer.state_dict(),
            "train_args": vars(self.args),
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

    def final_eval(self):
        # skip final eval if we are using a recipe
        if self.manager is None and self.checkpoint_manager is None:
            # patch the validator, so it always has access to the
            #  trainer object, which is needed to circumvent original ultralytics
            #  call that ignores the trainer object
            #  https://github.com/ultralytics/ultralytics/blob/
            #  6c65934b555e64bf26edd699865754b5ff651d0c/ultralytics/yolo/engine/trainer.py#L551
            self.validator = partial(self.validator, trainer=self)
            return super().final_eval()

    def callback_teardown(self):
        # NOTE: this callback is registered in __init__
        if self.manager is not None:
            self.manager.finalize()


class SparseDetectionTrainer(SparseTrainer, DetectionTrainer):
    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return SparseDetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
        )


class SparseClassificationTrainer(SparseTrainer, ClassificationTrainer):
    def get_validator(self):
        self.loss_names = ["loss"]
        return SparseClassificationValidator(self.test_loader, self.save_dir)


class SparseSegmentationTrainer(SparseTrainer, SegmentationTrainer):
    def get_validator(self):
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss"
        return SparseSegmentationValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
        )


class SparseYOLO(YOLO):
    def __init__(self, model="yolov8n.yaml", task="detect") -> None:
        model_str = str(model)

        if model_str.startswith("zoo:"):
            model = download_framework_model_by_recipe_type(
                Model(model_str), model_suffix="pt"
            )
            model_str = str(model)
            self.is_sparseml_checkpoint = True

        if model_str.endswith(".pt"):
            if os.path.exists(model_str):
                ckpt = torch.load(model_str, map_location="cpu")
                self.is_sparseml_checkpoint = (
                    "source" in ckpt and ckpt["source"] == "sparseml"
                )
            else:
                self.is_sparseml_checkpoint = False
        else:
            self.is_sparseml_checkpoint = False

        super().__init__(model, task)

        self.ModelClass = TASK_MAP[self.task][0]
        self.TrainerClass = TASK_MAP[self.task][1]
        self.ValidatorClass = TASK_MAP[self.task][2]
        self.PredictorClass = TASK_MAP[self.task][3]

        if self.TrainerClass == DetectionTrainer:
            self.TrainerClass = SparseDetectionTrainer
        elif self.TrainerClass == ClassificationTrainer:
            self.TrainerClass = SparseClassificationTrainer
        elif self.TrainerClass == SegmentationTrainer:
            self.TrainerClass = SparseSegmentationTrainer

        if self.ValidatorClass == DetectionValidator:
            self.ValidatorClass = SparseDetectionValidator
        elif self.ValidatorClass == ClassificationValidator:
            self.ValidatorClass = SparseClassificationValidator
        elif self.ValidatorClass == SegmentationValidator:
            self.ValidatorClass = SparseSegmentationValidator

    def _load(self, weights: str, task=None):
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
            self.ModelClass = TASK_MAP[self.task][0]

            if "yaml" in self.ckpt:
                self.model = self.ModelClass(dict(self.ckpt["yaml"]))
            elif "model_yaml" in self.ckpt:
                self.model = self.ModelClass(dict(self.ckpt["model_yaml"]))
            else:
                self.model = self.ModelClass(dict(self.ckpt["model"].yaml))

            if "recipe" in self.ckpt and self.ckpt["recipe"]:
                manager = ScheduledModifierManager.from_yaml(self.ckpt["recipe"])
                epoch = self.ckpt.get("epoch", -1)
                if epoch < 0:
                    epoch = float("inf")
                LOGGER.info(
                    "Applying structure from sparseml checkpoint "
                    f"at epoch {self.ckpt['epoch']}"
                )
                if manager.quantization_modifiers:
                    _modify_arch_for_quantization(self.model)
                manager.apply_structure(self.model, epoch=epoch)
            else:
                LOGGER.info("No recipe from in sparseml checkpoint")

            if self.ckpt["ema"]:
                self.model.load_state_dict(self.ckpt["ema"])
            else:
                self.model.load_state_dict(self.ckpt["model"])
            LOGGER.info("Loaded previous weights from checkpoint")
            assert self.model.yaml == self.ckpt["model_yaml"]

            self.overrides["model"] = weights
            self.overrides["task"] = self.task

        else:
            super()._load(weights, task)

    def export(self, **kwargs):
        """
        Export model.
        Args:
            **kwargs : Any other args accepted by the exporter.
        """
        if kwargs["imgsz"] is None:
            # if imgsz is not specified, remove it from the kwargs
            # so that it can be overridden by the model's default
            del kwargs["imgsz"]

        args = self.overrides.copy()
        args.update(kwargs)

        source = self.ckpt.get("source")
        recipe = self.ckpt.get("recipe")
        one_shot = args.get("one_shot")
        save_one_shot_torch = args.get("save_one_shot_torch")

        if source == "sparseml":
            LOGGER.info(
                "Source: 'sparseml' detected; "
                "Exporting model from SparseML checkpoint..."
            )
        else:
            LOGGER.info(
                "Source: 'sparseml' not detected; "
                "Exporting model from vanilla checkpoint..."
            )

        if one_shot:
            LOGGER.info(
                f"Detected one-shot recipe: {one_shot}. "
                "Applying it to the model to be exported..."
            )
            for p in self.model.parameters():
                p.requires_grad = True
            manager = ScheduledModifierManager.from_yaml(one_shot)

            overrides = self.overrides.copy()
            # assumes single-GPU or CPU one-shot pathway
            if kwargs["device"] is not None and "cpu" not in kwargs["device"]:
                overrides["device"] = "cuda:" + kwargs["device"]
            overrides["deterministic"] = kwargs["deterministic"]
            if kwargs["dataset_path"] is not None:
                overrides["data"] = data_from_dataset_path(
                    overrides["data"], kwargs["dataset_path"]
                )

            trainer = self.TrainerClass(overrides=overrides)
            self.model = self.model.to(trainer.device)

            manager.apply(
                self.model,
                # maybe we could check whether OBS pruner is in the manager?
                grad_sampler=create_grad_sampler(trainer, stride=32, model=self.model)
                if any(
                    map(
                        lambda mod: hasattr(mod, "_grad_sampler"),
                        manager.pruning_modifiers,
                    )
                )
                else None,
            )
            recipe = (
                ScheduledModifierManager.compose_staged(recipe, manager)
                if recipe
                else manager
            )

        save_dir = args["save_dir"]
        name = "model.onnx"  # save 'model.onnx' in deployment directory
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for _, m in self.model.named_modules():
            if isinstance(m, (Detect, Segment)):
                m.export = True

        # format attribute seems to not exist within this ultralytics update
        # This is a workaround. Should be one of
        # ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs')
        self.model.model[-1].format = "saved_model"
        exporter = ModuleExporter(self.model, save_dir)
        if save_one_shot_torch:
            if not one_shot:
                warnings.warn(
                    "No one-shot recipe detected; "
                    "skipping one-shot model torch export..."
                )
            else:
                torch_path = os.path.join(save_dir, name.replace(".onnx", ".pt"))
                LOGGER.info(f"Saving one-shot torch model to {torch_path}...")
                self.ckpt["model"] = self.model
                torch.save(self.ckpt, torch_path)

        dynamic_axes = kwargs.get("dynamic", {})
        if not dynamic_axes:
            dynamic_axes["output0"] = {0: "batch", 2: "anchors"}
            if isinstance(self, SegmentationModel):
                # set mask dimensions on output1 to dynamic
                dynamic_axes["output1"] = {
                    0: "batch",
                    2: "mask_height",
                    3: "mask_width",
                }

        exporter.export_onnx(
            sample_batch=torch.randn(1, 3, args["imgsz"], args["imgsz"]),
            opset=args["opset"],
            name=name,
            input_names=["images"],
            convert_qat=True,
            # ultralytics-specific argument
            do_constant_folding=True,
            output_names=["output0", "output1"]
            if isinstance(self.model, SegmentationModel)
            else ["output0"],
            dynamic_axes=dynamic_axes,
        )

        complete_path = os.path.join(save_dir, name)
        try:
            skip_onnx_input_quantize(complete_path, complete_path)
        except Exception:
            pass

        validate_onnx(complete_path)
        deployment_folder = exporter.create_deployment_folder(onnx_model_name=name)
        if args["export_samples"]:
            trainer_config = get_cfg(cfg=DEFAULT_SPARSEML_CONFIG_PATH)
            # First check if the yaml exists locally
            if os.path.exists(args["data"]):
                trainer_config.data = args["data"]
            else:
                # If it does not exist, may be a uralytics provided yaml. Try
                # downloading and updating path to dataset_path
                # Does this case actually happen? I.e. is args["data"] ever not a
                # checkpointed local yaml path?
                try:
                    if args["dataset_path"] is not None:
                        args["data"] = data_from_dataset_path(
                            args["data"], args["dataset_path"]
                        )
                        trainer_config.data = args["data"]
                except ValueError:
                    LOGGER.info(
                        f"yaml in {args['data']} could not be found. "
                        "Using default config"
                    )

            trainer_config.imgsz = args["imgsz"]
            trainer = DetectionTrainer(trainer_config)
            # inconsistency in name between
            # validation and test sets
            validation_set_path = trainer.testset
            device = trainer.device
            data_loader, _ = create_dataloader(
                path=validation_set_path, imgsz=args["imgsz"], batch_size=1, stride=32
            )

            export_sample_inputs_outputs(
                data_loader=data_loader,
                model=self.model,
                number_export_samples=args["export_samples"],
                device=device,
                save_dir=deployment_folder,
                onnx_path=os.path.join(deployment_folder, name),
            )

        if recipe:
            if isinstance(recipe, str):
                recipe = ScheduledModifierManager.from_yaml(recipe)

            LOGGER.info(
                "Recipe checkpoint detected, saving the "
                f"recipe to the deployment directory {deployment_folder}"
            )
            recipe.save(os.path.join(deployment_folder, "recipe.yaml"))

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

        overrides["nbs"] = int(overrides["nbs"])
        self.trainer = self.TrainerClass(overrides=overrides)
        if not overrides.get("resume"):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(
                weights=self.model if self.ckpt else None, cfg=self.model.yaml
            )
            self.model = self.trainer.model
        self.trainer.train()

    @smart_inference_mode()
    def val(self, data=None, **kwargs):
        overrides = self.overrides.copy()
        overrides["rect"] = True  # rect batches as default
        overrides.update(kwargs)
        overrides["mode"] = "val"
        if overrides.get("nbs"):
            overrides["nbs"] = int(overrides["nbs"])
        overrides["data"] = data or overrides["data"]
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.data = data or args.data
        args.task = self.task
        if args.imgsz == DEFAULT_CFG.imgsz:
            # use trained imgsz unless custom value is passed
            args.imgsz = self.ckpt["train_args"]["imgsz"]
        args.imgsz = check_imgsz(args.imgsz, max_dim=1)
        if args.task == "segment":
            args = check_coco128_segmentation(args)

        if not hasattr(self.model, "args"):
            # set model args from overrides if possible
            self.model.args = overrides

        validator = self.ValidatorClass(args=args)
        validator(
            model=self.model,
            trainer=self.TrainerClass(overrides=overrides),
            training=False,
        )


def generate_ddp_command(world_size, trainer):
    # NOTE: copied from ultralytics.yolo.utils.dist.generate_ddp_command
    """Generates and returns command for distributed training."""
    import __main__  # noqa local import to avoid https://github.com/Lightning-AI/lightning/issues/15218

    if not trainer.resume:
        shutil.rmtree(trainer.save_dir)  # remove the save_dir
    file = str(Path(sys.argv[0]).resolve())
    safe_pattern = re.compile(
        r"^[a-zA-Z0-9_. /\\-]{1,128}$"
    )  # allowed characters and maximum of 100 characters
    if not (
        safe_pattern.match(file) and Path(file).exists() and file.endswith(".py")
    ):  # using CLI
        file = generate_ddp_file(trainer)
    dist_cmd = "torch.distributed.run" if TORCH_1_9 else "torch.distributed.launch"
    port = find_free_network_port()
    cmd = [
        sys.executable,
        "-m",
        dist_cmd,
        "--nproc_per_node",
        f"{world_size}",
        "--master_port",
        f"{port}",
        file,
    ]
    return cmd, file


def generate_ddp_file(trainer):
    # NOTE: adapted from ultralytics.yolo.utils.dist.generate_ddp_file

    content = f"""if __name__ == "__main__":
    from sparseml.yolov8.trainers import {trainer.__class__.__name__}
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


def _get_submodule(module: torch.nn.Module, path: List[str]) -> torch.nn.Module:
    if not path:
        return module
    return _get_submodule(getattr(module, path[0]), path[1:])


def _modify_arch_for_quantization(model):
    layer_map = {"Bottleneck": Bottleneck, "Conv": Conv}
    for name, layer in model.named_modules():
        cls_name = layer.__class__.__name__
        if cls_name in layer_map:
            submodule_path = name.split(".")
            parent_module = _get_submodule(model, submodule_path[:-1])
            setattr(parent_module, submodule_path[-1], layer_map[cls_name](layer))
