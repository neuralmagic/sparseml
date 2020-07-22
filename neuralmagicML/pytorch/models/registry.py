"""
Code related to the PyTorch model registry for easily creating models.
"""

from typing import Union, List, Callable, Dict, Any, Tuple, NamedTuple
from merge_args import merge_args
from torch.nn import Module

from neuralmagicML.utils.frameworks import PYTORCH_FRAMEWORK
from neuralmagicML.utils import RepoModel, wrapper_decorator
from neuralmagicML.pytorch.utils import load_model


__all__ = ["ModelRegistry"]

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
        ("ignore_error_tensors", List[str]),
        ("args", Dict[str, Tuple[str, Any]]),
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
        load_strict: bool = True,
        ignore_error_tensors: List[str] = None,
        **kwargs
    ) -> Module:
        """
        Create a new model for the given key

        :param key: the model key (name) to create
        :param pretrained: True to load pretrained weights; to load a specific version
            give a string with the name of the version (recal, recal-perf), default None
        :param pretrained_path: A model file path to load into the created model
        :param pretrained_dataset: The dataset to load for the model
        :param load_strict: True to make sure all states are found in and
            loaded in model, False otherwise; default True
        :param ignore_error_tensors: tensors to ignore if there are errors in loading
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
            load_strict=load_strict,
            ignore_error_tensors=ignore_error_tensors,
            **kwargs
        )

    @staticmethod
    def create_repo(
        key: str, pretrained: Union[bool, str] = True, pretrained_dataset: str = None,
    ) -> RepoModel:
        """
        Create a RepoModel for the desired model in the model repo

        :param key: the model key (name) to retrieve
        :param pretrained: True to load pretrained weights; to load a specific version
            give a string with the name of the version (recal, recal-perf), default True
        :param pretrained_dataset: The dataset to load for the model
        :return: the RepoModel reference for the given model
        """
        if key not in ModelRegistry._CONSTRUCTORS:
            raise ValueError(
                "key {} is not in the model registry; available: {}".format(
                    key, ModelRegistry._CONSTRUCTORS
                )
            )

        attributes = ModelRegistry._ATTRIBUTES[key]

        return RepoModel(
            attributes.domain,
            attributes.sub_domain,
            attributes.architecture,
            attributes.sub_architecture,
            attributes.default_dataset
            if pretrained_dataset is None
            else pretrained_dataset,
            PYTORCH_FRAMEWORK,
            pretrained if isinstance(pretrained, str) else attributes.default_desc,
        )

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
        def_ignore_error_tensors: List[str] = None,
        desc_args: Dict[str, Tuple[str, Any]] = None,
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
        :param def_ignore_error_tensors: tensors to ignore if there are
            errors in loading
        :param desc_args: args that should be changed based on the description
        :return: the decorator
        """
        if not isinstance(key, List):
            key = [key]

        def decorator(const_func):
            const = ModelRegistry._registered_wrapper(key[0], const_func,)

            for r_key in key:
                if r_key in ModelRegistry._CONSTRUCTORS:
                    raise ValueError("key {} is already registered".format(key))

                ModelRegistry._CONSTRUCTORS[r_key] = const
                ModelRegistry._ATTRIBUTES[r_key] = _ModelAttributes(
                    input_shape,
                    domain,
                    sub_domain,
                    architecture,
                    sub_architecture,
                    default_dataset,
                    default_desc,
                    def_ignore_error_tensors,
                    desc_args,
                )

            return const

        return decorator

    @staticmethod
    def _registered_wrapper(
        key: str, const_func: Callable,
    ):
        @merge_args(const_func)
        @wrapper_decorator(const_func)
        def wrapper(
            pretrained_path: str = None,
            pretrained: Union[bool, str] = False,
            pretrained_dataset: str = None,
            load_strict: bool = True,
            ignore_error_tensors: List[str] = None,
            *args,
            **kwargs
        ):
            """
            :param pretrained_path: A path to the pretrained weights to load,
                if provided will override the pretrained param
            :param pretrained: True to load the default pretrained weights,
                a string to load a specific pretrained weight
                (ex: dense, recal, recal-perf),
                or False to not load any pretrained weights
            :param pretrained_dataset: The dataset to load pretrained weights for
                (ex: imagenet, mnist, etc).
                If not supplied will default to the one preconfigured for the model.
            :param load_strict: True to raise an error on issues with state dict
                loading from pretrained_path or pretrained, False to ignore
            :param ignore_error_tensors: Tensors to ignore while checking the state dict
                for weights loaded from pretrained_path or pretrained
            """
            attributes = ModelRegistry._ATTRIBUTES[key]

            if attributes.args and pretrained in attributes.args:
                kwargs[attributes.args[pretrained][0]] = attributes.args[pretrained][1]

            model = const_func(*args, **kwargs)
            ignore = []

            if ignore_error_tensors:
                ignore.extend(ignore_error_tensors)
            elif attributes.ignore_error_tensors:
                ignore.extend(attributes.ignore_error_tensors)

            if pretrained_path:
                load_model(pretrained_path, model, load_strict, ignore)
            elif pretrained:
                repo_model = ModelRegistry.create_repo(
                    key, pretrained, pretrained_dataset
                )
                try:
                    paths = repo_model.download_framework_files()
                    load_model(paths[0], model, load_strict, ignore)
                except Exception as ex:
                    # try one more time with overwrite on in case file was corrupted
                    paths = repo_model.download_framework_files(overwrite=True)
                    load_model(paths[0], model, load_strict, ignore)

            return model

        return wrapper
