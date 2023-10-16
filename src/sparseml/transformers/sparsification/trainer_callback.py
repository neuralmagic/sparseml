from transformers.trainer_callback import TrainerCallback as HFTrainerCallback

from sparseml.transformers.sparsification import TrainingArguments
from transformers.trainer_callback import TrainerState
from transformers import TrainerControl
from sparseml.core.session import callbacks

class TrainerCallback(HFTrainerCallback):
    """
    """

    def __init__(self, trainer: "RecipeManagerTrainerInterface", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainer = trainer

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of an epoch.
        """
        super().on_epoch_begin(args, state, control, **kwargs)
        callbacks.batch_start()


    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """
        super().on_epoch_end(args, state, control, **kwargs)
        callbacks.batch_end()

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        super().on_step_begin(args, state, control, **kwargs)
        callbacks.optim_pre_step()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        super().on_step_end(args, state, control, **kwargs)
        callbacks.optim_post_step()

    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        """
        Event called after a successful prediction.
        """
        super().on_predict(args, state, control, **kwargs)
        callbacks.loss_calculated()

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a prediction step.
        """
        pass