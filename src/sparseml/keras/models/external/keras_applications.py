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

"""
Module to register classification models from Keras applications
with the sparseml.keras model registry.
Models will have the key keras_applications.model_fn_name, e.g.,
keras_applications.MobileNetV2
"""

from inspect import getmembers, isfunction
from typing import Union

from sparseml import get_main_logger
from sparseml.keras.models.registry import ModelRegistry
from sparseml.keras.utils import keras


__all__ = []


_LOGGER = get_main_logger()


_supported_model_funcs = ["ResNet50"]


def _register_classification_models():
    # find model functions in keras.applications
    for model_func_name, model_func in getmembers(keras.applications, isfunction):
        if model_func_name not in _supported_model_funcs:
            continue
        key = "keras_applications.{}".format(model_func_name)
        input_shape = (
            (224, 224, 3) if model_func_name != "InceptionV3" else (299, 299, 3)
        )
        arch, sub_arch = _get_architecture(model_func_name)

        # wrap model constructor for registry compatibility
        wrapped_constructor = _registry_constructor_wrapper(key, model_func)

        ModelRegistry.register_wrapped_model_constructor(
            wrapped_constructor,
            key=key,
            input_shape=input_shape,
            domain="cv",
            sub_domain="classification",
            architecture=arch,
            sub_architecture=sub_arch,
            default_dataset="imagenet",
            default_desc="base",
            repo_source="keras.applications",
        )


def _registry_constructor_wrapper(key, model_func):
    # wraps the keras_applications model constructor function to be compatible
    # with sparseml model registry loading
    def wrapper(
        pretrained_path: str = None,
        pretrained: Union[bool, str] = False,
        pretrained_dataset: str = None,
        **kwargs,
    ):
        """
        :param pretrained_path: A path to the pretrained weights to load,
            if provided will override the pretrained param
        :param pretrained: True to load the default pretrained weights,
            a string to load a specific pretrained weight
            (ex: base, pruned-moderate),
            or False to not load any pretrained weights
        :param pretrained_dataset: The dataset to load pretrained weights for
            (ex: imagenet, mnist, etc).
            If not supplied will default to the one preconfigured for the model.
        """
        if isinstance(pretrained, str):
            if pretrained.lower() == "true":
                pretrained = True
            elif pretrained.lower() in ["false", "none"]:
                pretrained = False

        weights = None
        if pretrained:
            if pretrained_dataset == "imagenet":
                weights = "imagenet"  # Imagenet pretrained weights
            elif pretrained_path is not None:
                weights = pretrained_path  # Path to a weight file
        if weights is not None:
            _LOGGER.info("Model being created with weights from {}".format(weights))
        else:
            _LOGGER.info("Model being created with random weights")
        try:
            model = model_func(weights=weights, **kwargs)
        except ValueError:
            # Load the model directly assuming "weights" contain all information
            # about the model
            _LOGGER.info("Loading model directly from {}".format(weights))
            model = keras.models.load_model(weights)
        return model

    return wrapper


def _get_architecture(model_name):
    if model_name == "ResNet50":
        return "resnet_v1", "50"
    else:
        raise ValueError("Model {} unknown or not supported".format(model_name))


def main():
    _register_classification_models()


main()
