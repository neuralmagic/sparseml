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
Code related to the Keras model registry for easily creating models.
"""

from typing import Any, Callable, Dict, List, NamedTuple, Union

from merge_args import merge_args
from sparseml import get_main_logger
from sparseml.keras.utils import keras
from sparseml.utils import KERAS_FRAMEWORK, parse_optimization_str, wrapper_decorator
from sparsezoo import Model, model_args_to_stub


__all__ = [
    "ModelRegistry",
]


_LOGGER = get_main_logger()

"""
Simple named tuple object to store model info
"""
_ModelAttributes = NamedTuple(
    "_ModelAttributes",
    [
        ("input_shape", Any),
        ("domain", str),
        ("sub_domain", str),
        ("architecture", str),
        ("sub_architecture", str),
        ("default_dataset", str),
        ("default_desc", str),
        ("repo_source", str),
    ],
)


class ModelRegistry(object):
    """
    Registry class for creating models
    """

    _CONSTRUCTORS = {}  # type: Dict[str, Callable]
    _ATTRIBUTES = {}  # type: Dict[str, _ModelAttributes]

    @staticmethod
    def available_keys() -> List[str]:
        """
        :return: the keys (models) currently available in the registry
        """
        return list(ModelRegistry._CONSTRUCTORS.keys())

    @staticmethod
    def create(
        key: str,
        pretrained: Union[bool, str] = False,
        pretrained_path: str = None,
        pretrained_dataset: str = None,
        **kwargs,
    ) -> keras.Model:
        """
        Create a new model for the given key

        :param key: the model key (name) to create
        :param pretrained: True to load pretrained weights; to load a specific version
            give a string with the name of the version (pruned-moderate, base).
            Default None
        :param pretrained_path: A model file path to load into the created model
        :param pretrained_dataset: The dataset to load for the model
        :param kwargs: any keyword args to supply to the model constructor
        :return: the instantiated model
        """
        if key not in ModelRegistry._CONSTRUCTORS:
            raise ValueError(
                "key {} is not in the model registry; available: {}".format(
                    key, ModelRegistry._CONSTRUCTORS
                )
            )

        return ModelRegistry._CONSTRUCTORS[key](
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            pretrained_dataset=pretrained_dataset,
            **kwargs,
        )

    @staticmethod
    def create_zoo_model(
        key: str,
        pretrained: Union[bool, str] = True,
        pretrained_dataset: str = None,
    ) -> Model:
        """
        Create a sparsezoo Model for the desired model in the zoo

        :param key: the model key (name) to retrieve
        :param pretrained: True to load pretrained weights; to load a specific version
            give a string with the name of the version (optim, optim-perf), default True
        :param pretrained_dataset: The dataset to load for the model
        :return: the sparsezoo Model reference for the given model
        """
        if key not in ModelRegistry._CONSTRUCTORS:
            raise ValueError(
                "key {} is not in the model registry; available: {}".format(
                    key, ModelRegistry._CONSTRUCTORS
                )
            )

        attributes = ModelRegistry._ATTRIBUTES[key]

        sparse_name, sparse_category, sparse_target = parse_optimization_str(
            pretrained if isinstance(pretrained, str) else attributes.default_desc
        )

        model_dict = {
            "domain": attributes.domain,
            "sub_domain": attributes.sub_domain,
            "architecture": attributes.architecture,
            "sub_architecture": attributes.sub_architecture,
            "framework": KERAS_FRAMEWORK,
            "repo": attributes.repo_source,
            "dataset": attributes.default_dataset
            if pretrained_dataset is None
            else pretrained_dataset,
            "sparse_tag": f"{sparse_name}-{sparse_category}",
        }

        stub = model_args_to_stub(**model_dict)
        return Model(stub)

    @staticmethod
    def input_shape(key: str) -> Any:
        """
        :param key: the model key (name) to create
        :return: the specified input shape for the model
        """
        if key not in ModelRegistry._CONSTRUCTORS:
            raise ValueError(
                "key {} is not in the model registry; available: {}".format(
                    key, ModelRegistry._CONSTRUCTORS
                )
            )

        return ModelRegistry._ATTRIBUTES[key].input_shape

    @staticmethod
    def register(
        key: Union[str, List[str]],
        input_shape: Any,
        domain: str,
        sub_domain: str,
        architecture: str,
        sub_architecture: str,
        default_dataset: str,
        default_desc: str,
        repo_source: str = "sparseml",
    ):
        """
        Register a model with the registry. Should be used as a decorator

        :param key: the model key (name) to create
        :param input_shape: the specified input shape for the model
        :param domain: the domain the model belongs to; ex: cv, nlp, etc
        :param sub_domain: the sub domain the model belongs to;
            ex: classification, detection, etc
        :param architecture: the architecture the model belongs to;
            ex: resnet, mobilenet, etc
        :param sub_architecture: the sub architecture the model belongs to;
            ex: 50, 101, etc
        :param default_dataset: the dataset to use by default for loading
            pretrained if not supplied
        :param default_desc: the description to use by default for loading
            pretrained if not supplied
        :param repo_source: the source repo for the model, default is sparseml
        :return: the decorator
        """
        if not isinstance(key, List):
            key = [key]

        def decorator(const_func):
            wrapped_constructor = ModelRegistry._registered_wrapper(key[0], const_func)

            ModelRegistry.register_wrapped_model_constructor(
                wrapped_constructor,
                key,
                input_shape,
                domain,
                sub_domain,
                architecture,
                sub_architecture,
                default_dataset,
                default_desc,
                repo_source,
            )
            return wrapped_constructor

        return decorator

    @staticmethod
    def register_wrapped_model_constructor(
        wrapped_constructor: Callable,
        key: Union[str, List[str]],
        input_shape: Any,
        domain: str,
        sub_domain: str,
        architecture: str,
        sub_architecture: str,
        default_dataset: str,
        default_desc: str,
        repo_source: str,
    ):
        """
        Register a model with the registry from a model constructor or provider function

        :param wrapped_constructor: Model constructor wrapped to be compatible
            by call from ModelRegistry.create should have pretrained, pretrained_path,
            pretrained_dataset, load_strict, ignore_error_tensors, and kwargs as
            arguments
        :param key: the model key (name) to create
        :param input_shape: the specified input shape for the model
        :param domain: the domain the model belongs to; ex: cv, nlp, etc
        :param sub_domain: the sub domain the model belongs to;
            ex: classification, detection, etc
        :param architecture: the architecture the model belongs to;
            ex: resnet, mobilenet, etc
        :param sub_architecture: the sub architecture the model belongs to;
            ex: 50, 101, etc
        :param default_dataset: the dataset to use by default for loading
            pretrained if not supplied
        :param default_desc: the description to use by default for loading
            pretrained if not supplied
        :param repo_source: the source repo for the model; ex: sparseml, torchvision
        :return: The constructor wrapper registered with the registry
        """
        if not isinstance(key, List):
            key = [key]

        for r_key in key:
            if r_key in ModelRegistry._CONSTRUCTORS:
                raise ValueError("key {} is already registered".format(key))

            ModelRegistry._CONSTRUCTORS[r_key] = wrapped_constructor
            ModelRegistry._ATTRIBUTES[r_key] = _ModelAttributes(
                input_shape,
                domain,
                sub_domain,
                architecture,
                sub_architecture,
                default_dataset,
                default_desc,
                repo_source,
            )

    @staticmethod
    def _registered_wrapper(
        key: str,
        const_func: Callable,
    ):
        @merge_args(const_func)
        @wrapper_decorator(const_func)
        def wrapper(
            pretrained_path: str = None,
            pretrained: Union[bool, str] = False,
            pretrained_dataset: str = None,
            *args,
            **kwargs,
        ):
            """
            :param pretrained_path: A path to the pretrained weights to load,
                if provided will override the pretrained param
            :param pretrained: True to load the default pretrained weights,
                a string to load a specific pretrained weight
                (ex: base, optim, optim-perf),
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

            if pretrained_path:
                model = const_func(*args, **kwargs)
                try:
                    model.load_weights(pretrained_path)
                except ValueError:
                    _LOGGER.info("Loading model from {}".format(pretrained_path))
                    model = keras.models.load_model(pretrained_path)
            elif pretrained:
                zoo_model = ModelRegistry.create_zoo_model(
                    key, pretrained, pretrained_dataset
                )
                model_file_paths = zoo_model.download_framework_files(
                    extensions=[".h5"]
                )
                if not model_file_paths:
                    model_file_paths = zoo_model.download_framework_files(
                        extensions=[".tf"]
                    )
                if not model_file_paths:
                    raise RuntimeError("Error downloading model from SparseZoo")
                model_file_path = model_file_paths[0]
                model = keras.models.load_model(model_file_path)
            else:
                model = const_func(*args, **kwargs)
            return model

        return wrapper
