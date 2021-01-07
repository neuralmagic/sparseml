import numpy as np
import pytest
import tensorflow as tf
from sparseml.keras.optim import (
    ConstantPruningModifier,
    GMPruningModifer,
    MaskedLayer,
    PruningScheduler,
    ScheduledModifierManager,
    UnstructuredPruningMaskCreator,
)

from .mock import *


@pytest.mark.parametrize(
    "model_lambda, modifier_lambda",
    [
        (
            model_01,
            lambda: GMPruningModifer(
                params=["dense_01"],
                init_sparsity=0.25,
                final_sparsity=0.75,
                start_epoch=0.0,
                end_epoch=2.0,
                update_frequency=1.0,
            ),
        ),
        (
            model_01,
            lambda: GMPruningModifer(
                params=["dense_02"],
                init_sparsity=0.25,
                final_sparsity=0.75,
                start_epoch=0.0,
                end_epoch=2.0,
                update_frequency=1.0,
            ),
        ),
        (
            model_01,
            lambda: GMPruningModifer(
                params=["dense_01", "dense_02"],
                init_sparsity=0.25,
                final_sparsity=0.75,
                start_epoch=1.0,
                end_epoch=2.0,
                update_frequency=1.0,
            ),
        ),
    ],
    scope="function",
)
@pytest.mark.parametrize("steps_per_epoch", [10], scope="function")
class TestGradualKSModifier:
    def test_lifecycle(self, model_lambda, modifier_lambda, steps_per_epoch):
        model = model_lambda()
        modifier = modifier_lambda()
        loss = tf.keras.losses.categorical_crossentropy
        optimizer = tf.keras.optimizers.Adam()
        model, optimizer, step_callback = modifier.modify(
            model, optimizer, steps_per_epoch
        )
        epochs = 5
        batches = steps_per_epoch
        unused_arg = -1

        # Number of input and output neurons
        N = 100
        M = 10
        X_train = np.random.normal(size=(N, model.inputs[0].shape[1]))
        classes = np.random.randint(0, M, size=N)
        y_train = tf.keras.utils.to_categorical(classes, num_classes=M)

        step_callback.on_train_begin()
        for epoch in range(epochs):
            for batch in range(batches):
                step_callback.on_train_batch_begin(batch=unused_arg)
                with tf.GradientTape() as tape:
                    logits = model(X_train, training=True)
                    loss_value = loss(y_train, logits)
                    grads = tape.gradient(loss_value, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                step_callback.on_train_batch_end(batch=unused_arg)

                if epoch < modifier.start_epoch:
                    for layer in step_callback.prunable_layers:
                        assert not layer.mask_updater.update_ready

                if epoch == modifier.start_epoch and batch == 0:
                    for layer in step_callback.prunable_layers:
                        assert layer.mask_updater.update_ready
                        assert len(layer.pruning_vars) == 1
                        weight, mask, sparsity = layer.pruning_vars[0]
                        sparsity_val = tf.keras.backend.get_value(sparsity)
                        assert sparsity == modifier.init_sparsity

                if (
                    modifier.end_epoch > -1
                    and epoch == modifier.end_epoch - 1
                    and batch == batches - 1
                ):
                    for layer in step_callback.prunable_layers:
                        assert layer.mask_updater.update_ready
                        assert len(layer.pruning_vars) == 1
                        weight, mask, sparsity = layer.pruning_vars[0]
                        sparsity_val = tf.keras.backend.get_value(sparsity)
                        assert sparsity == modifier.final_sparsity
