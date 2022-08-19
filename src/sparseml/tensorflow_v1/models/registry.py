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
Code related to the PyTorch model registry for easily creating models.
"""


import re
from typing import Any, Callable, Dict, List, Optional, Union

from sparseml.tensorflow_v1.models.estimator import EstimatorModelFn
from sparseml.tensorflow_v1.utils import tf_compat
from sparseml.utils import TENSORFLOW_V1_FRAMEWORK, parse_optimization_str
from sparsezoo import Model, search_models


__all__ = ["ModelRegistry"]


class _ModelAttributes(object):
    def __init__(
        self,
        input_shape: Any,
        domain: str,
        sub_domain: str,
        architecture: str,
        sub_architecture: str,
        default_dataset: str,
        default_desc: str,
        default_model_fn_creator: EstimatorModelFn,
        base_name_scope: str,
        tl_ignore_tens: List[str],
        repo_source: str,
    ):
        self.input_shape = input_shape
        self.domain = domain
        self.sub_domain = sub_domain
        self.architecture = architecture
        self.sub_architecture = sub_architecture
        self.default_dataset = default_dataset
        self.default_desc = default_desc
        self.default_model_fn_creator = default_model_fn_creator
        self.base_name_scope = base_name_scope
        self.tl_ignore_tens = tl_ignore_tens
        self.repo_source = repo_source


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
    def create(key: str, *args, **kwargs) -> Any:
        """
        Create a new model for the given key

        :param key: the model key (name) to create
        :param args: any args to supply to the graph constructor
        :param kwargs: any keyword args to supply to the graph constructor
        :return: the outputs from the created graph
        """
        if key not in ModelRegistry._CONSTRUCTORS:
            raise ValueError(
                "key {} is not in the model registry; available: {}".format(
                    key, ModelRegistry._CONSTRUCTORS
                )
            )
        return ModelRegistry._CONSTRUCTORS[key](*args, **kwargs)

    @staticmethod
    def create_estimator(
        key: str,
        model_dir: str,
        model_fn_params: Optional[Dict[str, Any]],
        run_config: tf_compat.estimator.RunConfig,
        *args,
        **kwargs,
    ) -> tf_compat.estimator.Estimator:
        """
        Create Estimator for a model given the key and extra parameters

        :param key: the key that the model was registered with
        :param model_dir: directory to save results
        :param model_fn_params: parameters for model function
        :param run_config: RunConfig used by the estimator during training
        :param args: additional positional arguments to pass into model constructor
        :param kwargs: additional keyword arguments to pass into model constructor
        :return: an Estimator instance
        """
        model_const = ModelRegistry._CONSTRUCTORS[key]
        attributes = ModelRegistry._ATTRIBUTES[key]
        model_fn_creator = attributes.default_model_fn_creator()
        model_fn = model_fn_creator.create(model_const, *args, **kwargs)
        model_fn_params = {} if model_fn_params is None else model_fn_params
        classifier = tf_compat.estimator.Estimator(
            config=run_config,
            model_dir=model_dir,
            model_fn=model_fn,
            params=model_fn_params,
        )
        return classifier

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
            give a string with the name of the version (pruned-moderate, base),
            default True
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
            "framework": TENSORFLOW_V1_FRAMEWORK,
            "repo": attributes.repo_source,
            "dataset": attributes.default_dataset
            if pretrained_dataset is None
            else pretrained_dataset,
            "sparse_name": sparse_name,
            "sparse_category": sparse_category,
        }

        return search_models(**model_dict)[0]

    @staticmethod
    def load_pretrained(
        key: str,
        pretrained: Union[bool, str] = True,
        pretrained_dataset: str = None,
        pretrained_path: str = None,
        remove_dynamic_tl_vars: bool = False,
        sess: tf_compat.Session = None,
        saver: tf_compat.train.Saver = None,
    ):
        """
        Load pre-trained variables for a given model into a session.
        Uses a Saver object from TensorFlow to restore the variables
        from an index and data file.

        :param key: the model key (name) to create
        :param pretrained: True to load the default pretrained variables,
            a string to load a specific pretrained graph
            (ex: base, optim, optim-perf),
            or False to not load any pretrained weights
        :param pretrained_dataset: The dataset to load pretrained weights for
            (ex: imagenet, mnist, etc).
            If not supplied will default to the one preconfigured for the model.
        :param pretrained_path: A path to the pretrained variables to load,
            if provided will override the pretrained param
        :param remove_dynamic_tl_vars: True to remove the vars that are used for
            transfer learning (have a different shape and should not be restored),
            False to keep all vars in the Saver.
            Only used if saver is None
        :param sess: The session to load the model variables into
            if pretrained_path or pretrained is supplied.
            If not supplied and required, then will use the default session
        :param saver: The Saver instance to use to restore the variables
            for the graph if pretrained_path or pretrained is supplied.
            If not supplied and required, then will create one using the
            ModelRegistry.saver function
        """
        if key not in ModelRegistry._CONSTRUCTORS:
            raise ValueError(
                "key {} is not in the model registry; available: {}".format(
                    key, ModelRegistry._CONSTRUCTORS
                )
            )

        if not sess and (pretrained_path or pretrained):
            sess = tf_compat.get_default_session()

        if not saver and (pretrained_path or pretrained):
            saver = ModelRegistry.saver(key, remove_dynamic_tl_vars)

        if isinstance(pretrained, str):
            if pretrained.lower() == "true":
                pretrained = True
            elif pretrained.lower() in ["false", "none"]:
                pretrained = False

        if pretrained_path:
            saver.restore(sess, pretrained_path)
        elif pretrained:
            zoo_model = ModelRegistry.create_zoo_model(
                key, pretrained, pretrained_dataset
            )
            try:
                index_path = [
                    f.path
                    for f in zoo_model.training.files
                    if f.name.endswith(".index")
                ]
                index_path = index_path[0]
                model_path = index_path[:-6]
                saver.restore(sess, model_path)
            except Exception:
                # try one more time with overwrite on in case files were corrupted
                index_path = [
                    f.path
                    for f in zoo_model.training.files
                    if f.name.endswith(".index")
                ]

                if len(index_path) != 1:
                    raise FileNotFoundError(
                        "could not find .index file for {}".format(zoo_model.root_path)
                    )

                index_path = index_path[0]
                model_path = index_path[:-6]
                saver.restore(sess, model_path)

    @staticmethod
    def input_shape(key: str):
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
    def saver(key: str, remove_dynamic_tl_vars: bool = False) -> tf_compat.train.Saver:
        """
        Get a tf compat saver that contains only the variables for the desired
        architecture specified by key.
        Note, the architecture must have been created in the current graph already
        to work.

        :param key: the model key (name) to get a saver instance for
        :param remove_dynamic_tl_vars: True to remove the vars that are used for
            transfer learning (have a different shape and should not be restored),
            False to keep all vars in the Saver
        :return: a Saver object with the appropriate vars for the model to restore
        """
        if key not in ModelRegistry._CONSTRUCTORS:
            raise ValueError(
                "key {} is not in the model registry; available: {}".format(
                    key, ModelRegistry._CONSTRUCTORS
                )
            )
        base_name = ModelRegistry._ATTRIBUTES[key].base_name_scope
        saver_vars = [
            var
            for var in tf_compat.get_collection(tf_compat.GraphKeys.TRAINABLE_VARIABLES)
            if base_name in var.name
        ]
        saver_vars.extend(
            [
                var
                for var in tf_compat.global_variables()
                if ("moving_mean" in var.name or "moving_variance" in var.name)
                and base_name in var.name
            ]
        )

        if remove_dynamic_tl_vars:
            tl_ignore_tens = ModelRegistry._ATTRIBUTES[key].tl_ignore_tens

            def _check_ignore(var: tf_compat.Variable) -> bool:
                for ignore in tl_ignore_tens:
                    if re.match(ignore, var.name):
                        return True

                return False

            saver_vars = [var for var in saver_vars if not _check_ignore(var)]

        saver = tf_compat.train.Saver(saver_vars)

        return saver

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
        default_model_fn_creator: EstimatorModelFn,
        base_name_scope: str,
        tl_ignore_tens: List[str],
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
        :param default_model_fn_creator: default model creator to use when creating
            estimator instance
        :param base_name_scope: the base string used to create the graph under
        :param tl_ignore_tens: a list of tensors to ignore restoring for
            if transfer learning
        :param repo_source: the source repo for the model, default is sparseml
        :return: the decorator
        """
        if not isinstance(key, List):
            key = [key]

        def decorator(const_func):
            for r_key in key:
                if r_key in ModelRegistry._CONSTRUCTORS:
                    raise ValueError("key {} is already registered".format(key))

                ModelRegistry._CONSTRUCTORS[r_key] = const_func
                ModelRegistry._ATTRIBUTES[r_key] = _ModelAttributes(
                    input_shape,
                    domain,
                    sub_domain,
                    architecture,
                    sub_architecture,
                    default_dataset,
                    default_desc,
                    default_model_fn_creator,
                    base_name_scope,
                    tl_ignore_tens,
                    repo_source,
                )

            return const_func

        return decorator
