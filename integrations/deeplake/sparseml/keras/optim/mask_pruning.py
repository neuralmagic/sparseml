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

import abc
import collections
import inspect
from typing import List, Union

import tensorflow

from sparseml.keras.optim.mask_pruning_creator import (
    PruningMaskCreator,
    load_mask_creator,
)
from sparseml.keras.utils import keras


__all__ = [
    "MaskedLayer",
    "PruningScheduler",
    "remove_pruning_masks",
]


class PruningScheduler(abc.ABC):
    """
    Abstract pruning scheduler
    """

    _REGISTRY = {}

    def __init_subclass__(cls):
        super().__init_subclass__()
        PruningScheduler._register_class(cls)

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

    @abc.abstractmethod
    def get_config(self):
        raise NotImplementedError("Not implemented")

    @classmethod
    def deserialize(cls, config):
        """
        Deserialize a pruning scheduler from config returned by scheduler's
        get_config method

        :param config: a pruning scheduler's config
        :return: a pruning scheduler instance
        """
        if "class_name" not in config:
            raise ValueError("The 'class_name' not found in config: {}".format(config))
        class_name = config["class_name"]
        return keras.utils.deserialize_keras_object(
            config,
            module_objects=globals(),
            custom_objects={class_name: PruningScheduler._REGISTRY[class_name]},
        )

    @classmethod
    def _register_class(cls, target_cls):
        PruningScheduler._REGISTRY[target_cls.__name__] = target_cls


MaskedParamInfo = collections.namedtuple(
    "MaskedParamInfo", ["name", "param", "mask", "sparsity"]
)


class MaskAndWeightUpdater:
    """
    Core logic of updating masks and weights

    :param pruning_vars: a list of tuples where each element contains weight tensor,
        mask and sparsity
    :param pruning_scheduler: a pruning scheduler
    :param mask_creator: a mask creator
    :param global_step: a global step tensor
    """

    def __init__(
        self,
        pruning_vars: List[MaskedParamInfo],
        pruning_scheduler: PruningScheduler,
        mask_creator: PruningMaskCreator,
        global_step: tensorflow.Tensor,
    ):
        self._pruning_vars = pruning_vars
        self._pruning_scheduler = pruning_scheduler
        self._mask_creator = mask_creator
        self._global_step = global_step
        self._update_ready = None

    def _is_pruning_step(self) -> bool:
        global_step_val = keras.backend.get_value(self._global_step)
        assert global_step_val >= 0
        update_ready = self._pruning_scheduler.should_prune(global_step_val)
        return update_ready

    def _conditional_training_update(self):
        def _no_update_masks_and_weights():
            return tensorflow.no_op("no_update")

        def _update_masks_and_weights():
            assignments = []
            global_step_val = keras.backend.get_value(self._global_step)
            for masked_param_info in self._pruning_vars:
                new_sparsity = self._pruning_scheduler.target_sparsity(global_step_val)
                new_mask = self._mask_creator.create_sparsity_mask(
                    masked_param_info.param, new_sparsity
                )
                assignments.append(masked_param_info.mask.assign(new_mask))
                assignments.append(masked_param_info.sparsity.assign(new_sparsity))
                masked_param = tensorflow.math.multiply(
                    masked_param_info.param, masked_param_info.mask
                )
                assignments.append(masked_param_info.param.assign(masked_param))
            return tensorflow.group(assignments)

        update_ready = self._is_pruning_step()

        self._update_ready = update_ready
        return tensorflow.cond(
            tensorflow.cast(update_ready, tensorflow.bool),
            _update_masks_and_weights,
            _no_update_masks_and_weights,
        )

    def apply_masks(self):
        """
        Apply masks to the weights
        """
        assignments = []
        for masked_param_info in self._pruning_vars:
            masked_param = tensorflow.math.multiply(
                masked_param_info.param, masked_param_info.mask
            )
            assignments.append(masked_param_info.param.assign(masked_param))
        return tensorflow.group(assignments)

    def conditional_update(self, training=None):
        """
        Conditionally update masks and weights

        :param training: if in training mode
        """

        def _update():
            with tensorflow.control_dependencies([self._conditional_training_update()]):
                return tensorflow.no_op("update")

        def _no_update():
            return tensorflow.no_op("no_update")

        training = keras.backend.learning_phase() if training is None else training
        return tensorflow.cond(
            tensorflow.cast(training, tensorflow.bool), _update, _no_update
        )


_LAYER_PRUNABLE_PARAMS_MAP = {
    keras.layers.Conv1D: ["kernel"],
    keras.layers.Conv2D: ["kernel"],
    keras.layers.Conv2DTranspose: ["kernel"],
    keras.layers.Conv3D: ["kernel"],
    keras.layers.Conv3DTranspose: ["kernel"],
    keras.layers.Dense: ["kernel"],
    keras.layers.Embedding: ["embeddings"],
    keras.layers.LocallyConnected1D: ["kernel"],
    keras.layers.LocallyConnected2D: ["kernel"],
    keras.layers.SeparableConv1D: ["pointwise_kernel"],
    keras.layers.SeparableConv2D: ["pointwise_kernel"],
}


def _get_default_prunable_params(layer: keras.layers.Layer):
    if layer.__class__ in _LAYER_PRUNABLE_PARAMS_MAP:
        prunable_param_names = _LAYER_PRUNABLE_PARAMS_MAP[layer.__class__]
        return {
            "{}/{}".format(layer.name, param_name): getattr(layer, param_name)
            for param_name in prunable_param_names
        }
    else:
        expected_layers = [layer.__class__ for layer in _LAYER_PRUNABLE_PARAMS_MAP]
        raise ValueError(
            "Layer {} cannot be pruned. Expected layers: {}".format(
                layer, expected_layers
            )
        )


class MaskedLayer(keras.layers.Wrapper):
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
        layer: keras.layers.Layer,
        pruning_scheduler: PruningScheduler,
        mask_type: Union[str, List[int]] = "unstructured",
        **kwargs,
    ):
        if not isinstance(layer, MaskedLayer) and not isinstance(
            layer, keras.layers.Layer
        ):
            raise ValueError(
                "Invalid layer passed in, expected MaskedLayer or a keras Layer, "
                "but got {}".format(layer)
            )
        super(MaskedLayer, self).__init__(layer, **kwargs)
        self._layer = layer
        self._pruning_scheduler = pruning_scheduler
        self._mask_type = mask_type
        self._mask_creator = None
        self._pruning_vars = []
        self._global_step = None
        self._mask_updater = None

    def build(self, input_shape):
        super(MaskedLayer, self).build(input_shape)
        self._mask_creator = load_mask_creator(self._mask_type)
        self._pruning_vars = self._reuse_or_create_pruning_vars()
        self._global_step = self.add_weight(
            "global_step",
            shape=[],
            initializer=keras.initializers.Constant(-1),
            dtype=tensorflow.int64,
            trainable=False,
        )
        self._mask_updater = MaskAndWeightUpdater(
            self._pruning_vars,
            self._pruning_scheduler,
            self._mask_creator,
            self._global_step,
        )

    def _reuse_or_create_pruning_vars(
        self,
    ) -> List[MaskedParamInfo]:
        if isinstance(self._layer, MaskedLayer):
            # All nested masked layers reused pruning vars created
            # for the "core", inner-most, Keras built-in layer
            return self._layer.pruning_vars

        assert isinstance(self._layer, keras.layers.Layer)
        prunable_params = _get_default_prunable_params(self._layer)

        pruning_vars = []
        for name, param in prunable_params.items():
            mask = self.add_weight(
                "mask",
                shape=param.shape,
                initializer=keras.initializers.get("ones"),
                dtype=param.dtype,
                trainable=False,
            )
            sparsity = self.add_weight(
                "sparsity",
                shape=[],
                initializer=keras.initializers.get("zeros"),
                dtype=param.dtype,
                trainable=False,
            )
            pruning_vars.append(MaskedParamInfo(name, param, mask, sparsity))
        return pruning_vars

    def call(self, inputs: tensorflow.Tensor, training=None):
        """
        Forward function for calling layer instance as function
        """
        training = keras.backend.learning_phase() if training is None else training

        def _apply_masks_to_weights():
            with tensorflow.control_dependencies([self._mask_updater.apply_masks()]):
                return tensorflow.no_op("update")

        def _no_apply_masks_to_weights():
            return tensorflow.no_op("no_update_masks")

        tensorflow.cond(
            tensorflow.cast(training, tensorflow.bool),
            _apply_masks_to_weights,
            _no_apply_masks_to_weights,
        )

        args = inspect.getfullargspec(self._layer.call).args
        if "training" in args:
            return self._layer.call(inputs, training=training)
        else:
            return self._layer.call(inputs)

    def get_config(self):
        """
        Get layer config
        Serialization and deserialization should be done using
        keras.serialize/deserialize, which create and retrieve the "class_name"
        field automatically.
        The resulting config below therefore does not contain the field.
        """
        config = super(MaskedLayer, self).get_config()
        if "layer" not in config:
            raise RuntimeError("Expected 'layer' field not found in config")
        config.update(
            {
                "pruning_scheduler": self._pruning_scheduler.get_config(),
                "mask_type": self._mask_type,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        layer = keras.layers.deserialize(
            config.pop("layer"), custom_objects={"MaskedLayer": MaskedLayer}
        )
        if not isinstance(layer, MaskedLayer) and not isinstance(
            layer, keras.layers.Layer
        ):
            raise RuntimeError("Unexpected layer created from config")
        pruning_scheduler = PruningScheduler.deserialize(
            config.pop("pruning_scheduler")
        )
        if not isinstance(pruning_scheduler, PruningScheduler):
            raise RuntimeError("Unexpected pruning scheduler type created from config")
        mask_type = config.pop("mask_type")
        masked_layer = MaskedLayer(layer, pruning_scheduler, mask_type, **config)
        return masked_layer

    def compute_output_shape(self, input_shape):
        return self._layer.compute_output_shape(input_shape)

    @property
    def global_step(self):
        return self._global_step

    @property
    def mask_updater(self):
        return self._mask_updater

    @property
    def masks(self):
        return [masked_param_info.mask for masked_param_info in self._pruning_vars]

    @property
    def pruning_vars(self):
        return self._pruning_vars

    @property
    def pruned_layer(self):
        if isinstance(self._layer, MaskedLayer):
            return self._layer.pruned_layer
        elif isinstance(self._layer, keras.layers.Layer):
            return self._layer
        else:
            raise RuntimeError("Unrecognized layer")

    @property
    def masked_layer(self):
        return self._layer


def remove_pruning_masks(model: keras.Model):
    """
    Remove pruning masks from a model that was pruned using the MaskedLayer logic
    :param model: a model that was pruned using MaskedLayer
    :return: the original model with pruned weights
    """

    def _get_pruned_layer(layer):
        # If the model is loaded through SavedFormat, the layer of type
        # MaskedLayer would belong to a special package, hence the
        # second check below based simply on class name
        is_masked_layer = isinstance(
            layer, MaskedLayer
        ) or layer.__class__.__name__.endswith("MaskedLayer")
        if is_masked_layer:
            return _get_pruned_layer(layer.layer)
        elif isinstance(layer, keras.layers.Layer):
            return layer
        else:
            raise ValueError("Unknown layer type")

    def _remove_pruning_masks(layer):
        is_masked_layer = isinstance(
            layer, MaskedLayer
        ) or layer.__class__.__name__.endswith("MaskedLayer")
        if is_masked_layer:
            return _get_pruned_layer(layer)
        return layer

    # TODO: while the resulting model could be exported to ONNX, its built status
    # is removed
    return keras.models.clone_model(
        model, input_tensors=None, clone_function=_remove_pruning_masks
    )
