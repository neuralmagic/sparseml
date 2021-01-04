"""
Logging callbacks
"""

from typing import List
import tensorflow as tf

from sparseml.keras.optim.modifier import Modifier


class TensorBoardLoggingCallback(tf.keras.callbacks.TensorBoard):
    """
    Tensorboard logging callback

    :param modifiers: list of modifiers
    :param start_step_tensor: tensor containing the start step value
    :param log_dir: directory to log
    :param kwargs: optional, extra arguments passed into keras TensorBoard callback
    """
    def __init__(self, modifiers: List[Modifier], start_step_tensor: tf.Tensor, log_dir: str, **kwargs):
        super(TensorBoardLoggingCallback, self).__init__(log_dir=log_dir, **kwargs)
        self._modifiers = modifiers
        self._start_step_tensor = start_step_tensor
        self._step = None
        self._file_writer = tf.summary.create_file_writer(self.log_dir)

    def _log(self):
        """
        Retrieve logging values from modifiers and add to Tensorboard
        """
        log_data_vals = {}
        for mod in self._modifiers:
            log_data_vals.update(mod.add_summaries())

        with self._file_writer.as_default():
            for name, value in log_data_vals.items():
                tf.summary.scalar(name, value, step=self._step)
            self._file_writer.flush()

    def on_train_begin(self, logs=None):
        self._step = tf.keras.backend.get_value(self._start_step_tensor)

    def on_epoch_begin(self, epoch, logs=None):
        if logs is not None:
            super(TensorBoardLoggingCallback, self).on_epoch_begin(epoch, logs)
        if self.update_freq == "epoch":
            self._log()

    def _is_logging_step(self):
        return self.update_freq == "batch" or (
            isinstance(self.update_freq, int) and self._step % self.update_freq == 0
        )

    def on_train_batch_end(self, batch, logs=None):
        if logs is not None:
            super(TensorBoardLoggingCallback, self).on_train_batch_end(batch, logs)
        if self._is_logging_step():
            self._log()
        self._step += 1
