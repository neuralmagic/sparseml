"""
Code related to the pytorch model registry
"""

from typing import Union, List, Callable, Dict, Any, Tuple
from torch.nn import Module

from neuralmagicML.frameworks import PYTORCH_FRAMEWORK
from neuralmagicML.utils import RepoModel
from neuralmagicML.pytorch.utils import load_model


__all__ = ["ModelRegistry"]


class ModelRegistry(object):
    """
    Convenience class for getting and creating models
    """

    _CONSTRUCTORS = {}
    _INPUT_SHAPES = {}

    @staticmethod
    def create(
        key: str,
        pretrained: Union[bool, str] = False,
        pretrained_path: str = None,
        pretrained_dataset: str = None,
        load_strict: bool = True,
        ignore_error_tensors: List[str] = None,
        pre_load_func: Callable[[Module], Module] = None,
        post_load_func: Callable[[Module], Module] = None,
        *args,
        **kwargs
    ) -> Module:
        """
        Create a new model for the given key

        :param key: the model key (name) to create
        :param pretrained: True to load pretrained weights; to load a specific version
                           give input a string with the name of the version, default None
        :param pretrained_path: A model file path to load into the created model
        :param pretrained_dataset: The dataset to load for the model
        :param load_strict: True to make sure all states are found in and loaded in model,
                            False otherwise; default True
        :param ignore_error_tensors: tensors to ignore if there are errors in loading
        :param pre_load_func: a function to run before loading the pretrained weights
        :param post_load_func: a function to run after loading the pretrained weights
        :param args: any args to supply to the model constructor
        :param kwargs: any keyword args to supply to the model constructor
        :return: the instantiated model
        """
        if key not in ModelRegistry._CONSTRUCTORS:
            raise ValueError("key {} is not in the model registry")

        return ModelRegistry._CONSTRUCTORS[key](
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            pretrained_dataset=pretrained_dataset,
            load_strict=load_strict,
            ignore_error_tensors=ignore_error_tensors,
            pre_load_func=pre_load_func,
            post_load_func=post_load_func,
            *args,
            **kwargs
        )

    @staticmethod
    def input_shape(key: str):
        """
        :param key: the model key (name) to create
        :return: the specified input shape for the model
        """
        if key not in ModelRegistry._CONSTRUCTORS:
            raise ValueError("key {} is not in the model registry")

        return ModelRegistry._INPUT_SHAPES[key]

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
        :param sub_domain: the sub domain the model belongs to; ex: classification, detection, etc
        :param architecture: the architecture the model belongs to; ex: resnet, mobilenet, etc
        :param sub_architecture: the sub architecture the model belongs to; ex: 50, 101, etc
        :param default_dataset: the dataset to use by default for loading pretrained if not supplied
        :param default_desc: the description to use by default for loading pretrained if not supplied
        :param def_ignore_error_tensors: tensors to ignore if there are errors in loading
        :param desc_args: args that should be changed based on the description
        :return: the decorator
        """
        if not isinstance(key, List):
            key = [key]

        def decorator(const_func):
            const = ModelRegistry._registered_wrapper(
                const_func,
                domain,
                sub_domain,
                architecture,
                sub_architecture,
                default_dataset,
                default_desc,
                def_ignore_error_tensors,
                desc_args,
            )

            for r_key in key:
                if r_key in ModelRegistry._CONSTRUCTORS:
                    raise ValueError("key {} is already registered".format(key))

                ModelRegistry._CONSTRUCTORS[r_key] = const
                ModelRegistry._INPUT_SHAPES[r_key] = input_shape

            return const

        return decorator

    @staticmethod
    def _registered_wrapper(
        const_func: Callable,
        domain: str,
        sub_domain: str,
        architecture: str,
        sub_architecture: str,
        default_dataset: str,
        default_desc: str,
        def_ignore_error_tensors: List[str] = None,
        desc_args: Dict[str, Tuple[str, Any]] = None,
    ):
        def wrapper(
            pretrained: Union[bool, str] = False,
            pretrained_path: str = None,
            pretrained_dataset: str = None,
            load_strict: bool = True,
            ignore_error_tensors: List[str] = None,
            pre_load_func: Callable[[Module], Module] = None,
            post_load_func: Callable[[Module], Module] = None,
            *args,
            **kwargs
        ):
            if desc_args and pretrained in desc_args:
                kwargs[desc_args[pretrained][0]] = desc_args[pretrained[1]]

            model = const_func(*args, **kwargs)
            ignore = []

            if ignore_error_tensors:
                ignore.extend(ignore_error_tensors)
            elif def_ignore_error_tensors:
                ignore.extend(def_ignore_error_tensors)

            if pre_load_func:
                model = pre_load_func(model)

            if pretrained_path:
                load_model(pretrained_path, model, load_strict, ignore)
            elif pretrained:
                desc = pretrained if isinstance(pretrained, str) else default_desc
                dataset = pretrained_dataset if pretrained_dataset else default_dataset
                repo_model = RepoModel(
                    domain,
                    sub_domain,
                    architecture,
                    sub_architecture,
                    dataset,
                    PYTORCH_FRAMEWORK,
                    desc,
                )
                try:
                    path = repo_model.download_framework_file()
                    load_model(path, model, load_strict, ignore)
                except Exception as ex:
                    # try one more time with overwrite on in case file was corrupted
                    path = repo_model.download_framework_file(overwrite=True)
                    load_model(path, model, load_strict, ignore)

            if post_load_func:
                model = post_load_func(model)

            return model

        return wrapper
