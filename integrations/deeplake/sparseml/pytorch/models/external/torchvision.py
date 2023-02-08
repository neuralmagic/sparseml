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
Module to register torchvision imagenet classification models
with the sparseml.pytorch model registry.

Models will have the key torchvision.model_fn_name. ex: torchvision.mobilenet_v2

torchvision must be installed for models to register
"""

from inspect import getmembers, isfunction, signature
from typing import List, Union


try:
    from torchvision import models as torchvision_models
except Exception:
    torchvision_models = None

from sparseml.pytorch.models.registry import ModelRegistry
from sparseml.pytorch.utils import load_model


__all__ = []


def _register_classification_models():
    # find model functions in torchvision.models
    for model_name, constructor_function in getmembers(torchvision_models, isfunction):
        # using the "pretrained" keyword as proxy for a model function
        # NOTE: pretrained param was replaced with "weights" in torchvision 0.13.0
        params = signature(constructor_function).parameters
        if not ("pretrained" in params or "weights" in params):
            continue

        key = "torchvision.{}".format(model_name)
        image_size = (
            (3, 224, 224) if "inception_v3" not in model_name else (3, 299, 299)
        )
        arch, sub_arch = _get_architecture(model_name)

        # wrap model constructor for registry compatibility
        wrapped_constructor = _registry_constructor_wrapper(key, constructor_function)

        ModelRegistry.register_wrapped_model_constructor(
            wrapped_constructor,
            key=key,
            input_shape=image_size,
            domain="cv",
            sub_domain="classification",
            architecture=arch,
            sub_architecture=sub_arch,
            default_dataset="imagenet",
            default_desc="repo",
            repo_source="torchvision",
        )


def _registry_constructor_wrapper(key, constructor_function):
    # wraps the torchvision model constructor function to be compatible with sparseml
    # model registry loading
    def wrapper(
        pretrained_path: str = None,
        pretrained: Union[bool, str] = False,
        pretrained_dataset: str = None,
        load_strict: bool = True,
        ignore_error_tensors: List[str] = None,
        **kwargs,
    ):
        """
        :param pretrained_path: A path to the pretrained weights to load,
            if provided will override the pretrained param. May also be
            a SparseZoo stub path preceded by 'zoo:' with the optional
            `?recipe_type=` argument. If given a recipe type, the base
                model weights for that recipe will be loaded
        :param pretrained: True to load the default pretrained weights,
            a string to load a specific pretrained weight
            (ex: base, pruned-moderate),
            or False to not load any pretrained weights
        :param pretrained_dataset: The dataset to load pretrained weights for
            (ex: imagenet, mnist, etc).
            If not supplied will default to the one preconfigured for the model.
        :param load_strict: True to raise an error on issues with state dict
            loading from pretrained_path or pretrained, False to ignore
        :param ignore_error_tensors: Tensors to ignore while checking the state dict
            for weights loaded from pretrained_path or pretrained
        """
        if isinstance(pretrained, str):
            if pretrained.lower() == "true":
                pretrained = True
            elif pretrained.lower() in ["false", "none"]:
                pretrained = False

        pretrained_torchvision = pretrained is True and not pretrained_path
        model = constructor_function(pretrained=pretrained_torchvision, **kwargs)
        ignore_error_tensors = ignore_error_tensors or []

        if pretrained_path:
            load_model(pretrained_path, model, load_strict, ignore_error_tensors)
        elif pretrained and not pretrained_torchvision:
            zoo_model = ModelRegistry.create_zoo_model(
                key, pretrained, pretrained_dataset
            )
            try:
                paths = zoo_model.download_framework_files(extensions=[".pth"])
                load_model(paths[0], model, load_strict, ignore_error_tensors)
            except Exception:
                # try one more time with overwrite on in case file was corrupted
                paths = zoo_model.download_framework_files(
                    overwrite=True, extensions=[".pth"]
                )
                load_model(paths[0], model, load_strict, ignore_error_tensors)

        return model

    return wrapper


def _get_architecture(model_name):
    if "_v2_x" in model_name:  # shuffle net
        arch, sub_arch = model_name.split("_v2_x")
        sub_arch = ".".join(sub_arch.split("_")) + "x"
        return arch + "_v2", sub_arch

    if model_name == "googlenet":
        return "inception_v1", "googlenet"
    if model_name in ["alexnet", "inception_v3", "mobilenet_v2"]:
        return model_name, "none"

    if "mnasnet" in model_name or "squeezenet" in model_name:
        arch = "mnasnet" if "mnasnet" in model_name else "squeezenet"
        sub_arch = model_name.split(arch)[-1]
        sub_arch = ".".join(sub_arch.split("_")) if sub_arch != "1_0" else "none"
        return arch, sub_arch

    if model_name.startswith("resnext"):
        model_name = model_name.split("_")[0]
        sub_arch = model_name.split("resnext")[-1]
        return "resnext", sub_arch

    if model_name.startswith("wide_"):
        model_name = model_name.split("wide_")[-1]
        model_name += "xwidth"  # ie 2xwidth

    # split model name by location of first digit (ie resnet50, vgg11)
    for digit_idx, char in enumerate(model_name):
        if char.isdigit():
            break
    if digit_idx == len(model_name) - 1 and not model_name[-1].isdigit():
        arch = model_name
        sub_arch = "none"
    else:
        arch = model_name[0:digit_idx]
        sub_arch = model_name[digit_idx:]
    if arch == "resnet":
        arch += "_v1"
    return arch, sub_arch


if torchvision_models is not None:
    _register_classification_models()
