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

from transformers import TrainerControl
from transformers.trainer_callback import TrainerCallback as HFTrainerCallback
from transformers.trainer_callback import TrainerState

from sparseml.core.session import callbacks


class TrainerCallback(HFTrainerCallback):
    """ """

    def __init__(self, trainer: "RecipeManagerTrainerInterface", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainer = trainer

    def on_epoch_begin(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of an epoch.
        """
        super().on_epoch_begin(args, state, control, **kwargs)
        callbacks.batch_start()

    def on_epoch_end(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of an epoch.
        """
        super().on_epoch_end(args, state, control, **kwargs)
        callbacks.batch_end()

    # TODO check if step here beans batch or something else
    # might need to override the trainer function instead if we can't get the batch here
    # find where these hook into the training flow
    def on_step_begin(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of a training step. If using gradient
        accumulation, one training step might take several inputs.
        """
        super().on_step_begin(args, state, control, **kwargs)
        callbacks.optim_pre_step()

    def on_step_end(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of a training step. If using gradient accumulation,
        one training step might take several inputs.
        """
        super().on_step_end(args, state, control, **kwargs)
        callbacks.optim_post_step()

    def on_predict(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        metrics,
        **kwargs,
    ):
        """
        Event called after a successful prediction.
        """
        super().on_predict(args, state, control, **kwargs)
        # callbacks.loss_calculated()

    def on_prediction_step(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called after a prediction step.
        """
        pass
