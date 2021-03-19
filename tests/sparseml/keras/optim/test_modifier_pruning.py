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

import numpy as np
import pytest
import tensorflow

from sparseml.keras.optim import GMPruningModifier
from sparseml.keras.utils import keras
from tests.sparseml.keras.optim.mock import model_01


@pytest.mark.parametrize(
    "model_lambda, modifier_lambda",
    [
        (
            model_01,
            lambda: GMPruningModifier(
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
            lambda: GMPruningModifier(
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
            lambda: GMPruningModifier(
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
class TestGMPruningModifier:
    def test_lifecycle(self, model_lambda, modifier_lambda, steps_per_epoch):
        model = model_lambda()
        modifier = modifier_lambda()
        loss = keras.losses.categorical_crossentropy
        optimizer = keras.optimizers.Adam()
        model, optimizer, callbacks = modifier.modify(model, optimizer, steps_per_epoch)
        assert len(callbacks) == 1
        step_callback = callbacks[0]
        epochs = 5
        batches = steps_per_epoch
        unused_arg = -1

        # Number of input and output neurons
        N = 100
        M = 10
        X_train = np.random.normal(size=(N, model.inputs[0].shape[1]))
        classes = np.random.randint(0, M, size=N)
        y_train = keras.utils.to_categorical(classes, num_classes=M)

        step_callback.on_train_begin()
        for epoch in range(epochs):
            for batch in range(batches):
                step_callback.on_train_batch_begin(batch=unused_arg)
                with tensorflow.GradientTape() as tape:
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
                        assert sparsity == modifier.final_sparsity
