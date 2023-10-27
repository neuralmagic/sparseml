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

import collections
import inspect
import logging
import math
import os
import warnings
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.nn import Module
from transformers import Trainer as HFTransformersTrainer
from transformers import TrainerCallback, TrainerControl, TrainingArguments
from transformers.file_utils import PaddingStrategy
from transformers.integrations import TensorBoardCallback
from transformers.trainer_callback import TrainerState
from transformers.trainer_pt_utils import reissue_pt_warnings
from transformers.trainer_utils import get_last_checkpoint

import sparseml.core.session as sml
from sparseml.core import Recipe
from sparseml.core.framework import Framework
from sparseml.core.session import callbacks
from sparseml.pytorch.utils import (
    LoggerManager,
    ModuleSparsificationInfo,
    TensorBoardLogger,
)
from sparseml.transformers.finetune.helpers import _reload_model_state
from sparseml.transformers.utils.helpers import RECIPE_NAME


__all__ = [
    "SessionManagerMixIn",
    "TrainerInterface",
    "Trainer",
    "DisableHalfPrecisionCallback",
    "TransformersTrainer",
]

_LOGGER = logging.getLogger(__name__)
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class SessionManagerMixIn:
    def __init__(
        self,
        model: Module,
        model_state_path: str,
        recipe: Optional[str],
        pre_recipe_yaml: Optional[str],
        stripped_recipe: Optional[Recipe],
        recipe_args: Optional[Union[Dict[str, Any], str]] = None,
        metadata_args: Optional[List[str]] = None,
        data_args: Optional["DataTrainingArguments"] = None,  # noqa: F821
        teacher: Optional[Union[Module, str]] = None,
        **kwargs,
    ):
        # instantiate necessary state, like managers, so we can override args
        self.model = model
        self.model_state_path = str(model_state_path)
        self.recipe = recipe
        self.recipe_args = recipe_args
        self.teacher = teacher
        self.pre_recipe_yaml = pre_recipe_yaml
        self.stripped_recipe = stripped_recipe

        training_args = kwargs.get("args")
        self.metadata = (
            self._extract_metadata(
                metadata_args=metadata_args,
                training_args_dict=training_args.to_dict(),
                data_args_dict=asdict(data_args) if data_args else {},
            )
            if training_args and metadata_args
            else None
        )

        self.logger_manager = LoggerManager(log_python=False)
        sml.create_session()

        super().__init__(model=model, **kwargs)
        self.optim_callbacks = PostOptimCallback()
        self.callback_handler.add_callback(self.optim_callbacks)
        self.criterion = torch.nn.CrossEntropyLoss()
        self._add_tensorboard_logger_if_available()

        model_signature = inspect.signature(self.model.forward)
        self._model_signature_columns = list(model_signature.parameters.keys())

    def initialize_session(self, epoch: float, checkpoint: Optional[str]):
        orig_state_dict = self.model.state_dict()

        train_data = self.get_train_dataloader()

        sml.initialize(
            model=self.model,
            recipe=self.recipe,
            recipe_args=self.recipe_args,
            framework=Framework.pytorch,
            train_data=train_data,
            start=epoch,
            copy_data=False,
        )

        # reload the state dict for the model now that architecture matches expected
        # TODO: what if there is a quant modifier in the original recipe and we want to
        # continue adjusting its zero point and range?
        load_path = checkpoint or self.model_state_path
        if _reload_model_state(self.model, load_path, orig_state_dict):
            _LOGGER.info(
                "Reloaded model state after SparseML recipe structure modifications "
                f"from {load_path}"
            )

        return True

    def initialize_structure(self):
        session = sml.active_session()
        if session.lifecycle.initialized_ or session.lifecycle.pre_initialize_structure:
            return False

        sml.pre_initialize_structure()
        _LOGGER.info("Initialized SparseML structure from recipe argument")

    def finalize_session(self):
        session = sml.active_session()
        if not session.lifecycle.initialized_ or session.lifecycle.finalized:
            return False

        sml.finalize()
        _LOGGER.info("Finalized SparseML recipe argument applied to the model")

    def create_optimizer(self):
        """
        Override the optimizer to apply and update the recipe while training.
        create_optimizer must exist in the parent class and should set
        self.optimizer to the optimizer state and optionally set self.scaler
        if using amp.
        """

        self._check_super_defined("create_optimizer")
        super().create_optimizer()

        # n_gpu is already accounted for in the dataloader
        total_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
        )
        self.total_steps_per_epoch = math.ceil(
            len(self.train_dataset) / total_batch_size
        )
        sml.initialize(
            optimizer=self.optimizer, steps_per_epoch=self.total_steps_per_epoch
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

        # TODO: we don't currently have a LR scheduler in the new modifier framework
        self._check_super_defined("create_scheduler")
        if self.lr_scheduler is not None or sml.active_session() is None:
            super().create_scheduler(num_training_steps, optimizer)
            return

        # allow SparseML to manage LR and set a dummy scheduler
        # TODO: remove this and just using the HF one?
        self.lr_scheduler = self._dummy_lr_scheduler()

    def training_step(
        self, model: Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        self._check_super_defined("training_step")

        callbacks.batch_start(batch_data=inputs)
        model_outputs = super().training_step(model, inputs)

        return model_outputs

    def compute_loss(
        self, model: Module, inputs: Dict[str, Any], return_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Override for the compute_loss to factor trigger callbacks and filter columns

        :param model: the model to compute the loss for
        :param inputs: the inputs to pass through the model for calculating the loss
        :param return_outputs: True to return the outputs with the loss,
            False otherwise
        :return: the resulting loss if not return_outputs, otherwise a tuple
            containing the loss and the model's outputs
        """
        self._check_super_defined("compute_loss")

        inputs = {k: inputs[k] for k in inputs if k in self._model_signature_columns}
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs)
        callbacks.loss_calculated(loss=loss)
        callbacks.optim_pre_step()
        return loss

    def prediction_step(
        self,
        model: Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Wraps the prediction step from the original trainer to remove any input entry
        that should not be passed to the model.
        This situation may arise when distillation is used and the teacher model
        contains more inputs than the student model.
        """
        self._check_super_defined("prediction_step")

        inputs = {k: inputs[k] for k in inputs if k in self._model_signature_columns}

        model_outputs = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys
        )
        return model_outputs

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

        if sml.active_session() is None:
            return

        if output_dir is None:
            output_dir = self.args.output_dir

        recipe_path = os.path.join(output_dir, RECIPE_NAME)
        session = sml.active_session()
        recipe = session.lifecycle.recipe_container.compiled_recipe
        full_recipe = Recipe.simplify_combine_recipes([self.stripped_recipe, recipe])
        recipe_yaml_str = full_recipe.yaml()
        recipe_path = os.path.join(output_dir, "recipe.yaml")
        with open(recipe_path, "w") as fp:
            fp.write(recipe_yaml_str)

        _LOGGER.info(f"Saved SparseML recipe with model state to {recipe_path}")

        if self.pre_recipe_yaml is not None:
            pre_recipe_path = os.path.join(output_dir, "pre_recipe.yaml")
            with open(pre_recipe_path, "w") as fp:
                fp.write(self.pre_recipe_yaml)
        _LOGGER.info(f"Saved prior SparseML recipe to {pre_recipe_path}")

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
        self,
        metadata_args: List[str],
        training_args_dict: Dict[str, Any],
        data_args_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        metadata = {}
        if not training_args_dict.keys().isdisjoint(data_args_dict.keys()):
            raise ValueError(
                "Found common keys in `training_args` and `data args`. "
                "This is prohibitive and may lead to undesired behavior."
            )

        args_dict = {**training_args_dict, **data_args_dict}

        for arg in metadata_args:
            if arg not in args_dict.keys():
                logging.warning(
                    f"Required metadata argument {arg} was not found "
                    f"in the training arguments. Setting {arg} to None."
                )
                metadata[arg] = None
            else:
                metadata[arg] = args_dict[arg]

        return metadata

    def _check_super_defined(self, func: str):
        if not hasattr(super(), func):
            raise NotImplementedError(
                f"The super class for SparseMLTrainer must define a {func} function"
            )

    def _add_tensorboard_logger_if_available(self):
        tensorboard_callback = None
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, TensorBoardCallback):
                tensorboard_callback = callback
                break
        if tensorboard_callback is None:
            return

        if tensorboard_callback.tb_writer is None:
            tensorboard_callback._init_summary_writer(
                self.args, log_dir=self.args.logging_dir
            )

        self.logger_manager.add_logger(
            TensorBoardLogger(writer=tensorboard_callback.tb_writer)
        )

    def _get_fake_dataloader(
        self,
        num_samples: int,
        tokenizer: "PreTrainedTokenizerBase",  # noqa: F821
    ):
        # Rearrange inputs' keys to match those defined by model foward func, which
        # seem to define how the order of inputs is determined in the exported model
        forward_args_spec = inspect.getfullargspec(self.model.__class__.forward)
        synthetic_input = self._get_fake_input(
            forward_func_input_keys=forward_args_spec.args,
            tokenizer=tokenizer,
        )
        return (synthetic_input for _ in range(num_samples))

    def _get_fake_input(self, forward_func_input_keys, tokenizer):
        inputs = tokenizer(
            "", return_tensors="pt", padding=PaddingStrategy.MAX_LENGTH.value
        ).data  # Dict[Tensor]
        inputs = collections.OrderedDict(
            [
                (input_key, inputs[input_key][0].reshape(1, -1))
                for input_key in forward_func_input_keys
                if input_key in inputs
            ]
        )
        return inputs


class TrainerInterface(SessionManagerMixIn):
    """
    Training interface for running sparsification recipes with transformers flows.
    Mimics the lifecycle of transformers Trainer classes.

    Should be instantiated with multi-inheritance with a custom trainer class.
    TrainerInterface must be provided before Trainer for proper class dependency.
    i.e. class MyCustomTrainer(TrainerInterface, Trainer)

    :param model: the model to use with the trainer and apply sparsification to
    :param model_state_path: the state path to the model,
        used to load config and tokenizer settings
    :param recipe: the recipe, if any, to apply to the model and training
        process
    :param recipe_args: A json string, csv key=value string, or dictionary containing
        arguments to override the root arguments within the recipe such as
        learning rate or num epochs
    :param metadata_args A list of arguments to be extracted from training_args
        and passed as metadata for the final, saved recipe.
    :param teacher: teacher model for distillation. Set to 'self' to distill
        from the loaded model or 'disable' to turn off distillation
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
        applied = self.initialize_session(epoch=epoch, checkpoint=checkpoint)
        # self.callback_disable_fp16.check_disable(epoch, force=True)
        self.accelerator.wait_for_everyone()
        output = super().train(*args, **kwargs)
        self.accelerator.wait_for_everyone()
        if applied:
            self.finalize_session()

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            print("logging sparsification")
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
        self.initialize_structure()

        # Always evaluate w/ fp32 to be closer to DeepSparse
        use_cuda_amp = self.use_cuda_amp
        if not self.args.fp16_full_eval and not self.args.bf16_full_eval:
            self.use_cuda_amp = False

        output = super().evaluate(*args, **kwargs)
        self.use_cuda_amp = use_cuda_amp
        self.finalize_session()

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
        self.initialize_structure()
        output = super().predict(*args, **kwargs)
        self.finalize_session()

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


class TransformersTrainer(HFTransformersTrainer):
    """
    A transformers trainer class with custom behavior that can be shared
    by all trainers inside SparseML
    """

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


class PostOptimCallback(TrainerCallback):
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of a training step. If using gradient accumulation, one
        training step might take several inputs.
        """
        super().on_step_end(args, state, control, **kwargs)
        callbacks.optim_post_step()
        sml.callbacks.batch_end()

    def on_substep_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of an substep during gradient accumulation.
        """
        super().on_substep_end(args, state, control, **kwargs)
        callbacks.optim_post_step()
        sml.callbacks.batch_end()


class DisableHalfPrecisionCallback(TrainerCallback):
    """
    TrainerCallback for disabling FP16 training before QAT training begins
    :param sparseml_trainer: SparseML trainer that will call back into this object
    :param args: args to be passed to base TrainerCallback
    :param kwargs: key word arguments to be passed to base TrainerCallback
    """

    def __init__(self, trainer: SessionManagerMixIn, *args, **kwargs):
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
        # TODO: refactor to not use manager
        # would be nice to have helper functions for noting if quantization and/or
        # LR modifiers are applied
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
            self.trainer.use_cuda_amp = False

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
