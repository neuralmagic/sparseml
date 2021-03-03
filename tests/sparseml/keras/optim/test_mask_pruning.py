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

from typing import Dict, Union

import numpy as np
import pytest
import tensorflow as tf

from sparseml.keras.optim import (
    GMPruningModifier,
    MaskedLayer,
    ScheduledModifierManager,
)
from tests.sparseml.keras.optim.mock import (
    DenseLayerCreator,
    MockPruningScheduler,
    mnist_model,
)


@pytest.mark.parametrize(
    "layer_lambda, pruning_scheduler, mask_type, expected_mask",
    [
        (
            # Weight of a dense layer of shape (3, 4)
            DenseLayerCreator(
                "dense",
                np.array(
                    [[0.9, 0.1, 0.2, 0.5], [0.3, 0.5, 0.8, 0.9], [0.4, 0.6, 0.7, 0.9]]
                ),
            ),
            MockPruningScheduler([(1, 0.25), (2, 0.5)]),
            "unstructured",
            # List of expected mask, each corresponding to one of the
            # above update step in the MockPruningScheduler
            [
                # Expected mask at time step 1, 25% sparsity
                np.array(
                    [
                        [1, 0, 0, 1],
                        [0, 1, 1, 1],
                        [1, 1, 1, 1],
                    ]
                ),
                # Expected mask at time step 2, 50% sparsity
                np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 0, 1, 1],
                        [0, 1, 1, 1],
                    ]
                ),
            ],
        )
    ],
)
def test_mask_update_explicit(
    layer_lambda, pruning_scheduler, mask_type, expected_mask
):
    if tf.__version__ < "2":
        pytest.skip("Test needs to be fixed to run with tensorflow_v1 1.x")
    layer = layer_lambda()
    masked_layer = MaskedLayer(layer, pruning_scheduler, mask_type)
    masked_layer.build(input_shape=None)
    update_steps = list(pruning_scheduler.step_and_sparsity_pairs.keys())
    for idx, update_step in enumerate(update_steps):
        tf.keras.backend.batch_set_value([(masked_layer.global_step, update_step)])
        masked_layer.mask_updater.conditional_update(training=True)
        mask = tf.keras.backend.get_value(masked_layer.masks[0])
        assert np.allclose(mask, expected_mask[idx])


@pytest.mark.parametrize(
    "modifier_lambdas",
    [
        (
            lambda: GMPruningModifier(
                params=["conv2d/kernel:0"],
                init_sparsity=0.25,
                final_sparsity=0.75,
                start_epoch=0.0,
                end_epoch=2.0,
                update_frequency=1.0,
            ),
            lambda: GMPruningModifier(
                params=["conv2d_1/kernel:0"],
                init_sparsity=0.25,
                final_sparsity=0.75,
                start_epoch=0.0,
                end_epoch=2.0,
                update_frequency=1.0,
            ),
        ),
        (
            lambda: GMPruningModifier(
                params=["conv2d/kernel:0", "conv2d_2/kernel:0"],
                init_sparsity=0.25,
                final_sparsity=0.75,
                start_epoch=0.0,
                end_epoch=2.0,
                update_frequency=1.0,
            ),
            lambda: GMPruningModifier(
                params=["conv2d_1/kernel:0", "conv2d/kernel:0", "outputs/kernel:0"],
                init_sparsity=0.25,
                final_sparsity=0.75,
                start_epoch=2.0,
                end_epoch=3.0,
                update_frequency=1.0,
            ),
            lambda: GMPruningModifier(
                params=["conv2d_2/kernel:0", "outputs/kernel:0"],
                init_sparsity=0.25,
                final_sparsity=0.75,
                start_epoch=3.0,
                end_epoch=4.0,
                update_frequency=1.0,
            ),
        ),
    ],
    scope="function",
)
@pytest.mark.parametrize("steps_per_epoch", [10], scope="function")
def test_nested_layer_structure(modifier_lambdas, steps_per_epoch):
    model = mnist_model()
    modifiers = [mod() for mod in modifier_lambdas]
    manager = ScheduledModifierManager(modifiers)
    optimizer = tf.keras.optimizers.Adam()
    model, optimizer, callbacks = manager.modify(model, optimizer, steps_per_epoch)

    model.build(input_shape=(1, 28, 28, 1))

    # Verify number of (outer-most) masked layers
    modifier_masked_layer_names = [
        layer_name for mod in modifiers for layer_name in mod.layer_names
    ]
    model_masked_layer_names = [
        layer.name for layer in model.layers if isinstance(layer, MaskedLayer)
    ]
    assert len(model_masked_layer_names) == len(set(modifier_masked_layer_names))

    # Verify that if a layer is modified by N modifiers, then the corresponding
    # MaskedLayer will have N-1 number of MaskedLayer nested inside it
    for layer in model.layers:
        if isinstance(layer, MaskedLayer):
            expected_layers = len(
                [name for name in modifier_masked_layer_names if name == layer.name]
            )
            assert _count_nested_masked_layers(layer) == expected_layers

    # Verify the returned config dict has expected nested structures
    model_config = model.get_config()
    for layer_config in model_config["layers"]:
        if layer_config["class_name"] == "MaskedLayer":
            layer_name = layer_config["config"]["name"]
            expected_layers = len(
                [name for name in modifier_masked_layer_names if name == layer_name]
            )
            assert (
                _count_nested_masked_layers_in_config(layer_config) == expected_layers
            )

    # Verify model serialization and deserialization working for (nested) masked layer
    model_config = model.get_config()
    new_model = model.__class__.from_config(
        model_config, custom_objects={"MaskedLayer": MaskedLayer}
    )
    assert model_config == new_model.get_config()

    tf.keras.backend.clear_session()


def _count_nested_masked_layers_in_config(layer_config: Dict):
    if layer_config["class_name"] != "MaskedLayer":
        return 0
    return 1 + _count_nested_masked_layers_in_config(layer_config["config"]["layer"])


def _count_nested_masked_layers(layer: Union[MaskedLayer, tf.keras.layers.Layer]):
    if not isinstance(layer, MaskedLayer):
        return 0
    return 1 + _count_nested_masked_layers(layer.masked_layer)
