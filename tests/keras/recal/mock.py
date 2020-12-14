from typing import List, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from neuralmagicML.keras.recal import PruningScheduler

__all__ = [
    "DenseLayerCreator",
    "SequentialModelCreator",
    "MockPruningScheduler",
    "model_01",
]


class MockPruningScheduler(PruningScheduler):
    def __init__(self, step_and_sparsity_pairs: List[Tuple]):
        self.step_and_sparsity_pairs = {
            step: sparsity for (step, sparsity) in step_and_sparsity_pairs
        }

    def should_prune(self, step: int):
        return step in self.step_and_sparsity_pairs

    def target_sparsity(self, step: int):
        update_ready = step in self.step_and_sparsity_pairs
        sparsity = self.step_and_sparsity_pairs[step] if update_ready else None
        return sparsity


class DenseLayer(tf.keras.layers.Dense):
    def __init__(self, weight: np.ndarray):
        super(DenseLayer, self).__init__(weight.shape[1], activation=None)

        self.kernel = self.add_weight(
            "kernel",
            shape=weight.shape,
            initializer=tf.keras.initializers.Constant(weight),
            dtype=tf.float32,
            trainable=False,
        )
        self.bias = self.add_weight(
            "bias",
            shape=(weight.shape[1],),
            initializer=tf.keras.initializers.Constant(0.0),
            dtype=tf.float32,
            trainable=False,
        )


class LayerCreator:
    def __init__(self, kernel, bias=None):
        self.kernel = kernel
        self.bias = bias


class DenseLayerCreator(LayerCreator):
    def __init__(self, name, kernel, bias=None):
        super(DenseLayerCreator, self).__init__(kernel, bias)
        self.name = name

    def __call__(self, delay_build=False):
        layer = tf.keras.layers.Dense(
            self.kernel.shape[-1], activation=None, name=self.name
        )
        if not delay_build:
            layer.build((None, self.kernel.shape[0]))
            assert len(layer.get_weights()) == 2
            if self.bias is None:
                self.bias = np.zeros((self.kernel.shape[-1],))
            layer.set_weights([self.kernel, self.bias])
        return layer


class SequentialModelCreator:
    def __init__(self, layer_creators):
        self.layer_creators = layer_creators

    def __call__(self):
        model = tf.keras.Sequential()
        for layer_creator in self.layer_creators:
            model.add(layer_creator(delay_build=True))
        input_shape = tf.TensorShape((None, self.layer_creators[0].kernel.shape[-2]))
        model.build(input_shape=input_shape)
        return model


def model_01():
    inputs = tf.keras.Input(shape=(5))
    x = tf.keras.layers.Dense(10, name="dense_01")(inputs)
    outputs = tf.keras.layers.Dense(10, name="dense_02")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model
