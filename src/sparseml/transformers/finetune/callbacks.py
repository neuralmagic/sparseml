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


import logging
import math

from transformers import TrainerCallback, TrainerControl, TrainingArguments
from transformers.trainer_callback import TrainerState

import sparseml.core.session as session_manager


__all__ = [
    "DisableHalfPrecisionCallback",
    "TrainingLoopCallbacks",
]

_LOGGER = logging.getLogger(__name__)


class TrainingLoopCallbacks(TrainerCallback):
    """
    TrainerCallback for triggering SparseSession callbacks during the training loop.
    Used to update the model reference(for running with FSDP) and trigger the post-
    optim callbacks in each modifier.

    :param sparseml_trainer: SparseML trainer that will call back into this object
    :param args: args to be passed to base TrainerCallback
    :param kwargs: key word arguments to be passed to base TrainerCallback
    """

    def __init__(self, trainer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainer = trainer

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of training. Update the session reference to the
        model, as it will have changed to a wrapper if FSDP is enabled
        """
        super().on_train_begin(args, state, control, **kwargs)
        session = session_manager.active_session()
        session.state.model.model = self.trainer.model

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of a training step. If using gradient accumulation,
        one training step might take several inputs.

        Triggers optimizer post_step and batch_end in the active SparseSession
        """
        super().on_step_end(args, state, control, **kwargs)
        session_manager.callbacks.optim_post_step()
        session_manager.callbacks.batch_end()

    def on_substep_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of an substep during gradient accumulation.

        Triggers optimizer post_step and batch_end in the active SparseSession
        """
        super().on_substep_end(args, state, control, **kwargs)
        session_manager.callbacks.optim_post_step()
        session_manager.callbacks.batch_end()


class DisableHalfPrecisionCallback(TrainerCallback):
    """
    TrainerCallback for disabling FP16 training before QAT training begins

    :param sparseml_trainer: SparseML trainer that will call back into this object
    :param args: args to be passed to base TrainerCallback
    :param kwargs: key word arguments to be passed to base TrainerCallback
    """

    def __init__(self, trainer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainer = trainer
        self.on_begin_called = False
        self.quant_start_epoch = math.inf

    def check_disable(self, epoch: float, force: bool = False):
        """
        If needed due to active quantization, disable FP16 training
        """
        if (
            force or hasattr(self.trainer, "scaler") and self.trainer.scaler._enabled
        ) and self.qat_active():
            self.disable_amp(epoch)

    def qat_active(self) -> bool:
        """
        :return: True if a quantization modifier is active in the current session
        """
        session = session_manager.active_session()
        return session.state.model.qat_active()

    def disable_amp(self, epoch: float):
        """
        Disable FP16 training

        :param epoch: epoch to disable from
        """
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
        Event called at the beginning of an epoch. Disables FP16 training.
        """
        super().on_epoch_begin(args, state, control, **kwargs)
        self.on_begin_called = True
        self.check_disable(state.epoch)
