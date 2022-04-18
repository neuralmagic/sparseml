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
SparseML transformers trainer classes and interfaces to be plugged in with
existing or similiar HF trainer flows
"""

import inspect
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import torch
from torch import distributed as dist
from torch.nn import Module
from torch.utils.data import RandomSampler
from transformers import Trainer as TransformersTrainer
from transformers import TrainerCallback, TrainerControl, TrainingArguments
from transformers.file_utils import WEIGHTS_NAME
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import get_last_checkpoint

from sparseml.pytorch.optim import ScheduledModifierManager, ScheduledOptimizer
from sparseml.pytorch.utils import (
    GradSampler,
    LoggerManager,
    ModuleSparsificationInfo,
    WANDBLogger,
)
from sparseml.transformers.utils import SparseAutoModel
from sparseml.transformers.utils.helpers import RECIPE_NAME


__all__ = [
    "RecipeManagerTrainerInterface",
    "TrainerInterface",
    "Trainer",
    "DisableHalfPrecisionCallback",
]


_LOGGER = logging.getLogger(__name__)
TRAINER_STATE_NAME = "trainer_state.json"


class RecipeManagerTrainerInterface:
    """
    Training base interface for running sparsification recipes with transformers flows.
    Defines it's own lifecycle that is compatible with transformers flows.
    Can additionally be used outside of transformers flows provided
    they match reasonably closely.

    Should be instantiated with multi-inheritance with a custom trainer class.
    RecipeManagerTrainerInterface must be provided
    before Trainer for proper class dependency.
    i.e. class MyCustomTrainer(RecipeManagerTrainerInterface, Trainer)

    Expected lifecycle:
    1. apply_manager
    2. create_optimizer (only for training)
    3. create_scheduler (only for training)
    4. compute_loss (only for training, called before each step)
    5. save_model (only for training)
    6. finalize_manager

    :param model: the model to use with the trainer and apply sparsification to
    :param model_state_path: the state path to the model,
        used to load config and tokenizer settings
    :param recipe: the recipe, if any, to apply to the modle and training
        process
    :param recipe_args: A json string, csv key=value string, or dictionary containing
        arguments to override the root arguments within the recipe such as
        learning rate or num epochs
    :param metadata_args A list of arguments to be extracted from training_args
        and passed as metadata for the final, saved recipe.
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
        metadata_args: Optional[List[str]] = None,
        teacher: Optional[Union[Module, str]] = None,
        **kwargs,
    ):
        # instantiate necessary state, like managers, so we can override args
        self.model = model
        self.model_state_path = str(model_state_path)
        self.recipe = recipe
        self.recipe_args = recipe_args
        self.teacher = teacher
        self.metadata = (
            self._extract_metadata(
                metadata_args=metadata_args, training_args=kwargs["args"]
            )
            if ("args" in kwargs and metadata_args)
            else None
        )

        report_to = (
            ""
            if "args" not in kwargs
            or not kwargs["args"]
            or not kwargs["args"].report_to
            else kwargs["args"].report_to
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            loggers = [WANDBLogger()] if "wandb" in report_to else None
            if "modifier_log_frequency" in kwargs:
                self.logger_manager = LoggerManager(
                    loggers, log_frequency=kwargs["modifier_log_frequency"]
                )
            else:
                self.logger_manager = LoggerManager(loggers)
        else:
            self.logger_manager = LoggerManager(log_python=False)

        self.manager, self.arch_manager = self._setup_manager(kwargs)
        self.manager_applied = False
        self.manager_initialized = False
        self.manager_finalized = False
        self.manager_steps_per_epoch = 0

        super().__init__(model=model, **kwargs)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.callback_disable_fp16 = DisableHalfPrecisionCallback(self)
        self.callback_handler.add_callback(self.callback_disable_fp16)

        self.grad_sampler = GradSampler(
            self._mfac_data_loader(), self._mfac_loss_function
        )

        model_signature = inspect.signature(self.model.forward)
        self._model_signature_columns = list(model_signature.parameters.keys())

        if self.teacher is not None:
            teacher_signature = inspect.signature(self.teacher.forward)
            self._teacher_signature_columns = list(teacher_signature.parameters.keys())
        else:
            self._teacher_signature_columns = None

    def apply_manager(self, epoch: float, checkpoint: Optional[str]) -> bool:
        """
        Apply the recipe(s) to the model and training/validation process.

        :param epoch: the training epoch to apply the recipe(s) at.
            If loading after training, set epoch=math.inf
        :param checkpoint: the optional checkpoint to use to reload model state
            from after the model's architecture has been modified.
            If not supplied, falls back to self.model_state_path
        :return: True if recipes were applied, False otherwise
        """

        if (not self.arch_manager and self.manager is None) or self.manager_applied:
            return False

        orig_state_dict = self.model.state_dict()

        if self.arch_manager:
            self.arch_manager.apply_structure(self.model, epoch=epoch)
            _LOGGER.info(
                f"Applied structure from {self.arch_manager.num_stages()} "
                "previous recipe stage(s) to model and finalized "
                "(recipes saved with model_path)"
            )

        if self.manager is not None:
            self.manager.apply_structure(self.model, epoch=epoch)
            _LOGGER.info(
                "Applied structure from SparseML recipe argument to model at "
                f"epoch {epoch}"
            )

        # reload the state dict for the model now that architecture matches expected
        load_path = checkpoint or self.model_state_path
        self._reload_model_state(load_path, orig_state_dict)
        self.manager_applied = True
        _LOGGER.info(
            "Reloaded model state after SparseML recipe structure modifications "
            f"from {load_path}"
        )

        return True

    def finalize_manager(self) -> bool:
        """
        Finalize the current recipes to wrap up any held state.

        :return: True if recipes were finalized, False otherwise
        """
        if (
            self.manager is None
            or not self.manager_initialized
            or self.manager_finalized
        ):
            return False

        self.manager.finalize(self.model)
        self.manager_finalized = True
        _LOGGER.info("Finalized SparseML recipe argument applied to the model")

        return True

    def create_optimizer(self):
        """
        Override the optimizer to apply and update the recipe while training.
        create_optimizer must exist in the parent class and should set
        self.optimizer to the optimizer state and optionally set self.scaler
        if using amp.
        """
        self._check_super_defined("create_optimizer")
        super().create_optimizer()

        if not self.manager:
            return

        n_gpu = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else self.args._n_gpu
        )
        total_batch_size = (
            self.args.per_device_train_batch_size
            * n_gpu
            * self.args.gradient_accumulation_steps
        )
        self.manager_steps_per_epoch = math.ceil(
            len(self.train_dataset) / total_batch_size
        )

        if hasattr(self, "scaler"):
            wrap_optim_key = "scaler"
            self.scaler = self.manager.modify(
                self.model,
                self.optimizer,
                steps_per_epoch=self.manager_steps_per_epoch,
                allow_parallel_module=False,
                wrap_optim=self.scaler,
                loggers=self.logger_manager,
                distillation_teacher=self.teacher,
                grad_sampler=self.grad_sampler,
            )
        else:
            wrap_optim_key = "optimizer"
            self.optimizer = ScheduledOptimizer(
                self.optimizer,
                self.model,
                self.manager,
                steps_per_epoch=self.manager_steps_per_epoch,
                loggers=self.logger_manager,
                initialize_kwargs={
                    "grad_sampler": self.grad_sampler,
                    "distillation_teacher": self.teacher,
                },
            )
            if not self.manager.initialized:
                self.manager.initialize(
                    self.model,
                    loggers=self.logger_manager,
                    distillation_teacher=self.teacher,
                    grad_sampler=self.grad_sampler,
                )
        self.manager_initialized = True
        _LOGGER.info(
            f"Modified the {wrap_optim_key} from the recipe for training with "
            f"total_batch_size: {total_batch_size} and "
            f"steps_per_epoch: {self.manager_steps_per_epoch}"
        )

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ):
        """
        Create an LR scheduler to work with the applied recipes. If the
        recipe specifies LR modifiers, then will set lr_scheduler to a
        placeholder lr scheduler. Expects create_scheduler to be defined in the
        super class. Additionally expects self.lr_scheduler argument to be
        available

        :param num_training_steps: the total number of training steps
        :param optimizer: pre-initialized optimizer
        """
        self._check_super_defined("create_scheduler")

        if (
            self.lr_scheduler is not None
            or self.manager is None
            or not self.manager.learning_rate_modifiers
        ):
            super().create_scheduler(num_training_steps, optimizer)
            return

        # allow SparseML to manage LR and set a dummy scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            self.optimizer,
            lambda _: 1.0,
        )
        _LOGGER.warning("Overrode the lr_scheduler from SparseML recipe")

    def compute_loss(
        self, model: Module, inputs: Dict[str, Any], return_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Override for the compute_loss to factor in distillation modifiers.
        If distillation modifiers are present in the recipe, then will
        add the distillation loss to the normal loss function.
        Expects compute_loss to be defined in the suepr class.

        :param model: the model to compute the loss for
        :param inputs: the inputs to pass through the model for calculating the loss
        :param return_outputs: True to return the outputs with the loss,
            False otherwise
        :return: the resulting loss if not return_outputs, otherwise a tuple
            containing the loss and the model's outputs
        """
        self._check_super_defined("compute_loss")

        if (
            self.manager is None
            or not self.manager.initialized
            or not self.manager.enabled
            or not self.manager.distillation_modifiers
        ):
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

        student_inputs = {
            k: inputs[k] for k in inputs if k in self._model_signature_columns
        }
        student_outputs = model(**student_inputs)

        teacher_inputs = {
            k: inputs[k] for k in inputs if k in self._teacher_signature_columns
        }

        loss = student_outputs["loss"]
        loss = self.manager.loss_update(
            loss,
            model,
            self.optimizer,
            self.state.epoch,
            self.manager_steps_per_epoch,
            student_outputs=student_outputs,
            student_inputs=student_inputs,
            teacher_inputs=teacher_inputs,
        )

        return (loss, student_outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Wraps the prediction step from the original trainer to remove any input entry
        that should not be passed to model.
        This situation may arise when distillation is used and the teacher model
        contains more inputs than the student model.
        """
        self._check_super_defined("prediction_step")

        inputs = {k: inputs[k] for k in inputs if k in self._model_signature_columns}

        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    def save_model(
        self,
        output_dir: Optional[str] = None,
        _internal_call=True,
    ):
        """
        Override of the save_model function and expects it to exist in the parent.
        Calls into super() to save the model and additionally saves any recipes
        that were used with the model within the model folder.

        :param output_dir: the path to save the recipes into
        """
        """
        Save model during or after training. Modifiers that change the model
        architecture will also be saved
        """
        self._check_super_defined("save_model")
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)

        if self.manager is None:
            return

        if output_dir is None:
            output_dir = self.args.output_dir

        recipe_path = os.path.join(output_dir, RECIPE_NAME)
        if self.arch_manager:
            composed_manager = ScheduledModifierManager.compose_staged(
                base_recipe=str(self.arch_manager),
                additional_recipe=str(self.manager),
            )
            composed_manager.save(recipe_path)
        else:

            self.manager.save(recipe_path)

        _LOGGER.info(f"Saved SparseML recipe with model state to {recipe_path}")

    def log_model_sparsification(self):
        """
        Log the current model sparsification info including pruned and quantized states
        """
        sparsification_info = ModuleSparsificationInfo(self.model)

        _LOGGER.info(
            f"Sparsification info for {self.model_state_path}: "
            f"{sparsification_info.params_total} total params. "
            f"Of those there are {sparsification_info.params_prunable_total} prunable "
            f"params which have {sparsification_info.params_prunable_sparse_percent} "
            "avg sparsity."
        )
        model_type = (
            "sparse"
            if sparsification_info.params_prunable_sparse_percent > 5
            else "dense"
        )
        _LOGGER.info(
            f"{model_type} model detected, "
            f"all sparsification info: {sparsification_info}"
        )

    def _extract_metadata(
        self, metadata_args: List[str], training_args: TrainingArguments
    ) -> Dict:
        metadata = {}
        training_args_dict = training_args.to_dict()

        for arg in metadata_args:
            if arg not in training_args_dict.keys():
                logging.warning(
                    f"Required metadata argument {arg} was not found "
                    f"in the training arguments. Setting {arg} to None."
                )
                metadata[arg] = None
            else:
                metadata[arg] = training_args_dict[arg]

        return metadata

    def _check_super_defined(self, func: str):
        if not hasattr(super(), func):
            raise NotImplementedError(
                f"The super class for SparseMLTrainer must define a {func} function"
            )

    def _setup_manager(
        self, kwargs
    ) -> Tuple[Optional[ScheduledModifierManager], List[ScheduledModifierManager]]:
        manager = None
        arch_manager = None

        if self.recipe is not None:
            manager = ScheduledModifierManager.from_yaml(
                self.recipe,
                recipe_variables=self.recipe_args,
                metadata=self.metadata,
            )
            _LOGGER.info(
                "Loaded SparseML recipe variable into manager for recipe: "
                f"{self.recipe}, recipe_variables: {self.recipe_args} "
                f"and metadata {self.metadata}"
            )

        arch_recipe = os.path.join(self.model_state_path, RECIPE_NAME)
        if os.path.isfile(arch_recipe):
            arch_manager = ScheduledModifierManager.from_yaml(arch_recipe)
            _LOGGER.info(
                f"Loaded {arch_manager.num_stages()} SparseML checkpoint recipe "
                f"stage(s) from {arch_recipe} to replicate model sparse state"
            )

        if (
            manager is not None
            and manager.max_epochs
            and "args" in kwargs
            and (hasattr(kwargs["args"], "num_train_epochs"))
        ):
            _LOGGER.warning(
                f"Overriding num_train_epochs from Recipe to {manager.max_epochs}"
            )
            kwargs["args"].num_train_epochs = manager.max_epochs

        return manager, arch_manager

    def _reload_model_state(self, load_path: str, orig_state_dict: Dict[str, Any]):
        if (
            not load_path
            or not os.path.isdir(load_path)
            or not os.path.isfile(os.path.join(load_path, WEIGHTS_NAME))
        ):
            _LOGGER.warning(
                "Model state was not reloaded for SparseML: "
                f"could not find model wieghts for model_path {load_path}"
            )
            return

        current_state_dict = self.model.state_dict()

        if set(orig_state_dict.keys()) == set(current_state_dict):
            # no change in keys, ignore reload
            return

        # change in keys due to architecture changes, reload statedict
        load_state_dict = torch.load(
            os.path.join(load_path, WEIGHTS_NAME), map_location="cpu"
        )
        _, missing, unexpected, _, _ = self.model._load_state_dict_into_model(
            self.model, load_state_dict, load_path, _fast_init=False
        )

        if missing:
            _LOGGER.warning(
                "Missing keys found when reloading model state for SparseML recipe:"
                f"{missing}"
            )

        if unexpected:
            _LOGGER.warning(
                f"Unexpected keys found when reloading model state for SparseML recipe:"
                f"{unexpected}"
            )

        total_loaded = len(current_state_dict) - (len(missing) if len(missing) else 0)
        _LOGGER.info(
            f"Reloaded {total_loaded} model params for SparseML Recipe from {load_path}"
        )
        SparseAutoModel.log_model_load(
            self.model,
            self.model_state_path,
            model_type="student" if self.teacher else "model",
            delayed_load=False,
        )

    def _mfac_data_loader(self):
        data_loader_template = self.get_train_dataloader()

        data_loader = torch.utils.data.DataLoader(
            dataset=data_loader_template.dataset,
            batch_size=data_loader_template.batch_size // 2,
            sampler=RandomSampler(data_loader_template.dataset, replacement=False),
            num_workers=data_loader_template.num_workers,
            collate_fn=data_loader_template.collate_fn,
            pin_memory=data_loader_template.pin_memory,
            drop_last=data_loader_template.drop_last,
            timeout=data_loader_template.timeout,
            worker_init_fn=data_loader_template.worker_init_fn,
            generator=data_loader_template.generator,
            prefetch_factor=data_loader_template.prefetch_factor,
            persistent_workers=data_loader_template.persistent_workers,
        )

        for sample in data_loader:
            if self.label_smoother is not None and "labels" in sample:
                label = sample.pop("labels")
            else:
                label = None
            sample = self._prepare_inputs(sample)
            yield [], sample, label

    def _mfac_loss_function(self, model_outputs, loss_target):
        if loss_target is not None:
            loss = self.label_smoother(model_outputs, loss_target)
        else:
            loss = (
                model_outputs["loss"]
                if isinstance(model_outputs, dict)
                else model_outputs[0]
            )
        return loss


class TrainerInterface(RecipeManagerTrainerInterface):
    """
    Training interface for running sparsification recipes with transformers flows.
    Mimics the lifecycle of transformers Trainer classes.

    Should be instantiated with multi-inheretance with a custom trainer class.
    TrainerInterface must be provided before Trainer for proper class dependency.
    i.e. class MyCustomTrainer(TrainerInterface, Trainer)

    :param model: the model to use with the trainer and apply sparsification to
    :param model_state_path: the state path to the model,
        used to load config and tokenizer settings
    :param recipe: the recipe, if any, to apply to the modle and training
        process
    :param recipe_args: A json string, csv key=value string, or dictionary containing
        arguments to override the root arguments within the recipe such as
        learning rate or num epochs
    :param metadata_args A list of arguments to be extracted from training_args
        and passed as metadata for the final, saved recipe.
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
        metadata_args: Optional[List[str]] = None,
        teacher: Optional[Union[Module, str]] = None,
        **kwargs,
    ):

        super().__init__(
            model=model,
            model_state_path=model_state_path,
            recipe=recipe,
            recipe_args=recipe_args,
            metadata_args=metadata_args,
            teacher=teacher,
            **kwargs,
        )

    def train(self, *args, **kwargs):
        """
        Run a sparsification training cycle.
        Calls into apply_manager before super().train()
        and calls finalize_manager, if applied, after super().train().

        :param args: positional args to pass to super().train()
        :param kwargs: keyword args to pass to super().train()
        :return: the output from super.train()
        """
        checkpoint, epoch = self._generate_apply_manager_params(kwargs)
        applied = self.apply_manager(epoch=epoch, checkpoint=checkpoint)
        self.callback_disable_fp16.check_disable(epoch, force=True)
        output = super().train(*args, **kwargs)
        if applied:
            self.finalize_manager()
        self.log_model_sparsification()

        return output

    def evaluate(self, *args, **kwargs):
        """
        Run a sparsification evaluation cycle.
        Calls into apply_manager before super().evaluate()
        and calls finalize_manager, if applied, after super().evaluate().

        :param args: positional args to pass to super().evaluate()
        :param kwargs: keyword args to pass to super().evaluate()
        :return: the output from super.evaluate()
        """
        applied = self.apply_manager(epoch=math.inf, checkpoint=None)
        output = super().evaluate(*args, **kwargs)
        if applied:
            self.finalize_manager()

        return output

    def predict(self, *args, **kwargs):
        """
        Run a sparsification prediction cycle.
        Calls into apply_manager before super().predict()
        and calls finalize_manager, if applied, after super().predict().

        :param args: positional args to pass to super().predict()
        :param kwargs: keyword args to pass to super().predict()
        :return: the output from super.predict()
        """
        applied = self.apply_manager(epoch=math.inf, checkpoint=None)
        output = super().predict(*args, **kwargs)
        if applied:
            self.finalize_manager()

        return output

    def _generate_apply_manager_params(self, kwargs) -> Tuple[Optional[str], float]:
        checkpoint = None
        epoch = 0.0

        if not kwargs or "resume_from_checkpoint" not in kwargs:
            _LOGGER.warning(
                "resume_from_checkpoint not passed into SparseMLTrainer.train. "
                "This will cause issues with restoring recipes when "
                "running from a checkpoint."
            )
        elif kwargs["resume_from_checkpoint"]:
            if (
                isinstance(kwargs["resume_from_checkpoint"], bool)
                and kwargs["resume_from_checkpoint"]
            ):
                checkpoint = get_last_checkpoint(self.args.output_dir)
            else:
                checkpoint = kwargs["resume_from_checkpoint"]
            epoch = TrainerState.load_from_json(
                os.path.join(checkpoint, TRAINER_STATE_NAME)
            ).epoch

        return checkpoint, epoch


class Trainer(TrainerInterface, TransformersTrainer):
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

    def _remove_unused_columns(
        self, dataset: "datasets.Dataset", description: Optional[str] = None
    ):
        if not self.args.remove_unused_columns:
            return dataset
        if self._signature_columns is None and self.teacher is not None:
            model_signature = inspect.signature(self.model.forward)
            model_signature_columns = set(model_signature.parameters.keys())

            teacher_signature = inspect.signature(self.teacher.forward)
            teacher_signature_columns = set(teacher_signature.parameters.keys())

            self._signature_columns = list(
                model_signature_columns | teacher_signature_columns
            )

            # Labels may be named label or label_ids, the default data
            # collator handles that.
            self._signature_columns += ["label", "label_ids"]

        return super()._remove_unused_columns(dataset, description)


class DisableHalfPrecisionCallback(TrainerCallback):
    """
    TrainerCallback for disabling FP16 training before QAT training begins
    :param sparseml_trainer: SparseML trainer that will call back into this object
    :param args: args to be passed to base TrainerCallback
    :param kwargs: key word arguments to be passed to base TrainerCallback
    """

    def __init__(self, trainer: RecipeManagerTrainerInterface, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainer = trainer
        self.on_begin_called = False
        self.quant_start_epoch = math.inf

    def check_disable(self, epoch: float, force: bool = False):
        if (
            force or hasattr(self.trainer, "scaler") and self.trainer.scaler._enabled
        ) and self.qat_active(epoch):
            self.disable_amp(epoch)

    def qat_active(self, epoch: float) -> bool:
        manager_q_active = arch_manager_q_active = False
        if self.trainer.manager:
            manager_q_active = bool(self.trainer.manager.qat_active(epoch))
        if self.trainer.arch_manager:
            arch_manager_q_active = bool(
                self.trainer.arch_manager.quantization_modifiers
            )
        return manager_q_active or arch_manager_q_active

    def disable_amp(self, epoch: float):
        if not self.on_begin_called:
            # disable if training loops haven't started so we don't load
            # the empty scaler state dict and instead disable it from the start
            self.trainer.use_amp = False

        if hasattr(self.trainer, "scaler"):
            self.trainer.scaler._enabled = False

        self.quant_start_epoch = epoch
        _LOGGER.info(f"entering QAT phase at epoch {epoch}, disabling FP16 training")

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
        self.on_begin_called = True
        self.check_disable(state.epoch)

        if state.epoch > self.quant_start_epoch:
            _LOGGER.info(self.trainer.model)
