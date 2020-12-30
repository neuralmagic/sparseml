"""
Masking Keras layers to support pruning, and logics behind mask and weight updates
"""

import abc
import inspect
from typing import List, Tuple
import tensorflow as tf

from sparseml.keras.optim.mask_pruning_creator import PruningMaskCreator

__all__ = ["MaskedLayer", "PruningScheduler"]


class PruningScheduler(abc.ABC):
    """
    Abstract pruning scheduler
    """

    @abc.abstractmethod
    def should_prune(self, step: int) -> bool:
        """
        Check if the given step is a right time for pruning

        :param step: training step
        :return: True if pruning should take place; False otherwise
        """
        raise NotImplementedError("Not implemented")

    @abc.abstractmethod
    def target_sparsity(self, step: int, **kwargs) -> float:
        """
        Compute the target sparsity at the given step

        :param step: training step
        :param kwargs: optional keyword params that a specific scheduler might need
        :return: target sparsity
        """
        raise NotImplementedError("Not implemented")


class MaskAndWeightUpdater:
    """
    Core logic of updating masks and weights

    :param pruning_vars: a list of tuples where each element contains weight tensor, mask and sparsity
    :param pruning_scheduler: a pruning scheduler
    :param mask_creator: a mask creator
    :param global_step: a global step tensor
    """

    def __init__(
        self,
        pruning_vars: List[Tuple[tf.Tensor, tf.Tensor, tf.Tensor]],
        pruning_scheduler: PruningScheduler,
        mask_creator: PruningMaskCreator,
        global_step: tf.Tensor,
    ):
        self.pruning_vars = pruning_vars
        self.pruning_scheduler = pruning_scheduler
        self.mask_creator = mask_creator
        self.global_step = global_step
        self.update_ready = None

    def _is_pruning_step(self) -> bool:
        global_step_val = tf.keras.backend.get_value(self.global_step)
        assert global_step_val >= 0
        update_ready = self.pruning_scheduler.should_prune(global_step_val)
        return update_ready

    def _conditional_training_update(self):
        def _no_update_masks_and_weights():
            return tf.no_op("no_update")

        def _update_masks_and_weights():
            assignments = []
            global_step_val = tf.keras.backend.get_value(self.global_step)
            for weight, mask, sparsity in self.pruning_vars:
                new_sparsity = self.pruning_scheduler.target_sparsity(global_step_val)
                new_mask = self.mask_creator.create_sparsity_mask(weight, new_sparsity)
                assignments.append(mask.assign(new_mask))
                assignments.append(sparsity.assign(new_sparsity))
                masked_weight = tf.math.multiply(weight, mask)
                assignments.append(weight.assign(masked_weight))
            return tf.group(assignments)

        update_ready = self._is_pruning_step()
        self.update_ready = update_ready
        return tf.cond(
            tf.cast(update_ready, tf.bool),
            _update_masks_and_weights,
            _no_update_masks_and_weights,
        )

    def conditional_update(self, training=None):
        """
        Conditionally update masks and weights

        :param training: if in training mode
        """

        def _update():
            with tf.control_dependencies([self._conditional_training_update()]):
                return tf.no_op("update")

        def _no_update():
            return tf.no_op("no_update")

        training = tf.keras.backend.learning_phase() if training is None else training
        return tf.cond(tf.cast(training, tf.bool), _update, _no_update)


def _get_default_prunable_weights(layer: tf.keras.layers.Layer):
    if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(
        layer, tf.keras.layers.Dense
    ):
        return [layer.kernel]
    else:
        raise ValueError("Expected Conv2D, Dense layers, but got {}".format(layer))


class MaskedLayer(tf.keras.layers.Wrapper):
    """
    Masked layer is a layer wrapping around another layer with a mask; the mask however
    is shared if the enclosed layer is again of MaskedLayer type

    :param layer: either a MaskedLayer or a keras layer
    :param pruning_scheduler: a pruning scheduler
    :param mask_creator: a mask creator
    :param kwargs: optional params for keras layer constructor, e.g. layer name
    """

    def __init__(
        self,
        layer: tf.keras.layers.Layer,
        pruning_scheduler: PruningScheduler,
        mask_creator: PruningMaskCreator,
        **kwargs
    ):
        if not isinstance(layer, MaskedLayer) and not isinstance(
            layer, tf.keras.layers.Layer
        ):
            raise ValueError(
                "Invalid layer passed in, expected MaskedLayer or a keras Layer, but got {}".format(
                    layer
                )
            )
        super(MaskedLayer, self).__init__(layer, **kwargs)
        self._layer = layer
        self._pruning_scheduler = pruning_scheduler
        self._mask_creator = mask_creator
        self._global_step = self.add_weight(
            "global_step",
            shape=[],
            initializer=tf.keras.initializers.Constant(-1),
            dtype=tf.int64,
            trainable=False,
        )
        self._pruning_vars = self._reuse_or_create_pruning_vars()
        self._mask_updater = MaskAndWeightUpdater(
            self._pruning_vars,
            self._pruning_scheduler,
            self._mask_creator,
            self._global_step,
        )

    def _reuse_or_create_pruning_vars(
        self,
    ) -> List[Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
        if isinstance(self._layer, MaskedLayer):
            # All nested masked layers reused pruning vars created
            # for the "core", inner-most, Keras built-in layer
            return self._layer.pruning_vars

        assert isinstance(self._layer, tf.keras.layers.Layer)
        if hasattr(self._layer, "get_prunable_weights"):
            # A layer subclassed from PrunableLayer
            prunable_weights = self._layer.get_prunable_weights()
        else:
            prunable_weights = _get_default_prunable_weights(self._layer)

        pruning_vars = []
        for weight in prunable_weights:
            mask = self.add_weight(
                "mask",
                shape=weight.shape,
                initializer=tf.keras.initializers.get("ones"),
                dtype=weight.dtype,
                trainable=False,
            )
            sparsity = self.add_weight(
                "sparsity",
                shape=[],
                initializer=tf.keras.initializers.get("zeros"),
                dtype=weight.dtype,
                trainable=False,
            )
            pruning_vars.append((weight, mask, sparsity))
        return pruning_vars

    def call(self, inputs: tf.Tensor, training=None):
        """
        Forward function for calling layer instance as function
        """
        training = tf.keras.backend.learning_phase() if training is None else training
        args = inspect.getfullargspec(self._layer.call).args
        if "training" in args:
            return self._layer.call(inputs, training=training)
        else:
            return self._layer.call(inputs)

    @property
    def global_step(self):
        return self._global_step

    @property
    def mask_updater(self):
        return self._mask_updater

    @property
    def masks(self):
        return [m for (_, m, _) in self._pruning_vars]
