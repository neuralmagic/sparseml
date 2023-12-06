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

import inspect
import logging
import math
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel,
    StateDictType,
)
from torch.nn import Module
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import get_last_checkpoint

import sparseml.core.session as session_manager
from sparseml.core.framework import Framework
from sparseml.core.session import callbacks
from sparseml.pytorch.model_load.helpers import reload_model_state
from sparseml.pytorch.utils import LoggerManager, ModuleSparsificationInfo
from sparseml.transformers.finetune.callbacks import (
    DisableHalfPrecisionCallback,
    TrainingLoopCallbacks,
)
from sparseml.transformers.utils.helpers import RECIPE_NAME


__all__ = [
    "SessionManagerMixIn",
]

_LOGGER = logging.getLogger(__name__)
TRAINER_STATE_NAME = "trainer_state.json"


class SessionManagerMixIn:
    """
    Mix-In class to extend the Hugging Face Trainer class to support SparseML recipes
    for one-shot and finetuning flows.

    :param model: PyTorch model to run training on
    :param model_state_path: path to Pytorch model checkpoint or saved model
    :param recipe: path to recipe file to apply during training
    :param recipe_args: additional kwargs to use for evaluating recipe
    :param metadata_args: additional kwargs for configuring training
    :param data_args: kwargs for configuring dataset loading
    :param teacher: optional teacher model to use for distillation
    """

    def __init__(
        self,
        model: Module,
        model_state_path: str,
        recipe: Optional[str],
        recipe_args: Optional[Union[Dict[str, Any], str]] = None,
        metadata_args: Optional[List[str]] = None,
        data_args: Optional["DataTrainingArguments"] = None,  # noqa: F821
        teacher: Optional[Union[Module, str]] = None,
        **kwargs,
    ):
        # instantiate necessary state, like managers, so we can override args
        self.model_state_path = str(model_state_path)
        self.recipe = recipe
        self.recipe_args = recipe_args
        self.teacher = teacher

        # parse training and metadata args
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

        # setup logger and session
        self.logger_manager = LoggerManager(log_python=False)
        session_manager.create_session()

        # call Trainer initialization
        super().__init__(model=model, **kwargs)

        # setup callbacks and loss
        self.optim_callbacks = TrainingLoopCallbacks(self)
        self.callback_handler.add_callback(self.optim_callbacks)
        self.callback_disable_fp16 = DisableHalfPrecisionCallback(self)
        self.callback_handler.add_callback(self.callback_disable_fp16)
        self.criterion = torch.nn.CrossEntropyLoss()

        model_signature = inspect.signature(model.forward)
        self._model_signature_columns = list(model_signature.parameters.keys())

        if self.teacher is not None and teacher not in ("disable", "self"):
            teacher_signature = inspect.signature(self.teacher.forward)
            self._teacher_signature_columns = list(teacher_signature.parameters.keys())
        else:
            self._teacher_signature_columns = None

    def initialize_session(self, epoch: float, checkpoint: Optional[str]):
        """
        Initialize the SparseSession from the specified epoch, evaluates the recipe
        and initialized the modifiers for the training session

        :param epoch: Epoch to initialize session from, usually 0 unless loading
        from a checkpoint
        :param checkpoint: Optional checkpoint to initialize from to continue training
        """
        session = session_manager.active_session()
        if session.lifecycle.initialized_ or session.lifecycle.finalized:
            return False

        orig_state_dict = self.model.state_dict()
        train_data = self.get_train_dataloader()

        self.accelerator.wait_for_everyone()
        session_manager.initialize(
            model=self.model,
            teacher_model=self.teacher,  # TODO: what about for self/disable?
            recipe=self.recipe,
            recipe_args=self.recipe_args,
            framework=Framework.pytorch,
            train_data=train_data,
            start=epoch,
            copy_data=False,
            fsdp_active=self.is_fsdp_enabled,
        )
        self.accelerator.wait_for_everyone()

        # reload the state dict for the model now that architecture matches expected
        # TODO: what if there is a quant modifier in the original recipe and we want to
        # continue adjusting its zero point and range?
        load_path = checkpoint or self.model_state_path
        if reload_model_state(self.model, load_path, orig_state_dict):
            _LOGGER.info(
                "Reloaded model state after SparseML recipe structure modifications "
                f"from {load_path}"
            )

    def initialize_structure(self):
        """
        Initialize any recipe structural changes such as quantization on the model,
        return immediately if structure or session has already been initialized
        """
        session = session_manager.active_session()
        if session.lifecycle.initialized_ or session.lifecycle.pre_initialize_structure:
            return False

        session_manager.pre_initialize_structure(
            model=self.model,
            recipe=self.recipe,
            recipe_args=self.recipe_args,
            framework=Framework.pytorch,
        )
        _LOGGER.info(f"Initialized SparseML structure from recipe {self.recipe}")

    def finalize_session(self):
        """
        Wrap up training by finalizing all modifiers initialized in the current session
        """
        session = session_manager.active_session()
        if not session.lifecycle.initialized_ or session.lifecycle.finalized:
            return False

        with FullyShardedDataParallel.summon_full_params(self.model):
            # in order to update each layer we need to gathers all its parameters
            session_manager.finalize()
        _LOGGER.info("Finalized SparseML session")

    def create_optimizer(self):
        """
        Override the optimizer to apply and update the recipe while training.
        create_optimizer must exist in the parent class and should set
        self.optimizer to the optimizer state and optionally set self.scaler
        if using amp.
        """

        self._check_super_defined("create_optimizer")
        super().create_optimizer()

        # n_gpu handled internally by dataloader
        total_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
        )
        self.total_steps_per_epoch = math.ceil(
            len(self.train_dataset) / total_batch_size
        )
        session_manager.initialize(
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
        if self.lr_scheduler is not None or session_manager.active_session() is None:
            super().create_scheduler(num_training_steps, optimizer)
            return

        # allow SparseML to manage LR and set a dummy scheduler
        # TODO: remove this and just using the HF one?
        self.lr_scheduler = self._dummy_lr_scheduler()

    def training_step(
        self, model: Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Overrides the Trainer's training step to trigger the batch_start callback to
        the modifiers, then calls the parent function.

        :param model: the model to compute the loss for
        :param inputs: the inputs to pass through the model for calculating the loss
        :return: output of the model
        """
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

        # TODO: do we need these model signature columns?
        inputs = {k: inputs[k] for k in inputs if k in self._model_signature_columns}
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs)

        if session_manager.active_session().lifecycle.initialized_:
            state = callbacks.loss_calculated(loss=loss)
            if state.loss is not None:
                loss = state.loss
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

    def train(self, *args, **kwargs):
        """
        Run a sparsification training cycle. Runs initialization for the sparse session
        before calling super().train() and finalization of the session after.

        Logs sparsification details for the trained model.

        :param args: positional args to pass to super().train()
        :param kwargs: keyword args to pass to super().train()
        :return: the output from super.train()
        """
        checkpoint, epoch = self._calculate_checkpoint_info(kwargs)
        self.initialize_session(epoch=epoch, checkpoint=checkpoint)
        self.callback_disable_fp16.check_disable(epoch, force=True)
        self.accelerator.wait_for_everyone()
        output = super().train(*args, **kwargs)
        self.accelerator.wait_for_everyone()
        self.finalize_session()

        self.accelerator.wait_for_everyone()

        # Need to gather parameters across the GPUs before accessing layer weights
        with FullyShardedDataParallel.summon_full_params(self.model):
            self.log_model_sparsification()

        return output

    def evaluate(self, *args, **kwargs):
        """
        Run a sparsification evaluation cycle.
        Runs initialize_structure for the sparse session before calling
        super().evaluate() and finalization of the session after.

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
        Runs initialize_structure for the sparse session before calling
        super().predict() and finalization of the session after.

        :param args: positional args to pass to super().predict()
        :param kwargs: keyword args to pass to super().predict()
        :return: the output from super.predict()
        """
        self.initialize_structure()
        output = super().predict(*args, **kwargs)
        self.finalize_session()

        return output

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
        self._check_super_defined("save_model")
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)

        if session_manager.active_session() is None:
            return  # nothing to save

        if output_dir is None:
            output_dir = self.args.output_dir

        full_state_dict_config = FullStateDictConfig(
            offload_to_cpu=True, rank0_only=True
        )

        if isinstance(self.model, FullyShardedDataParallel):
            with FullyShardedDataParallel.state_dict_type(
                self.model, StateDictType.FULL_STATE_DICT, full_state_dict_config
            ):
                state_dict = self.accelerator.get_state_dict(self.model, unwrap=False)

            self.accelerator.unwrap_model(self.model).save_pretrained(
                output_dir,
                is_main_process=self.accelerator.is_main_process,
                save_function=self.accelerator.save,
                state_dict=state_dict,
            )

        self.save_state()
        self.save_optimizer_and_scheduler(output_dir)

        # save recipe, will contain modifiers from the model's original recipe as well
        # as those added from self.recipe
        recipe_path = os.path.join(output_dir, RECIPE_NAME)
        session = session_manager.active_session()
        recipe = session.lifecycle.recipe_container.compiled_recipe
        recipe_yaml_str = recipe.yaml()
        recipe_path = os.path.join(output_dir, "recipe.yaml")
        with open(recipe_path, "w") as fp:
            fp.write(recipe_yaml_str)

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
                f"The super class for SessionManagerMixIn must define a {func} function"
            )

    def _calculate_checkpoint_info(self, kwargs) -> Tuple[Optional[str], float]:
        """
        If resuming from checkpoint is set, get checkpoint and epoch to resume from
        """
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
