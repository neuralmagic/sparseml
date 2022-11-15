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

__all__ = [
    "create_recipe",
    "create_sparse_transfer_recipe",
    "create_pruning_recipe",
    "create_quantization_recipe",
]

from typing import Optional

from sparseml.pytorch import recipe_template


def create_recipe(
    model: Optional["Module"] = None,  # noqa: F821
    pruning: str = "true",
    quant: bool = True,
    lr_func: str = "linear",
    **recipe_args,
) -> str:
    """
    Convenience function to create a recipe based on supplied args and kwargs

    :param model: an instantiated PyTorch Module, or the local path to a torch.jit
        loadable *.pt file, if supplied then the recipe is built according to this
        architecture
    :param pruning: optional pruning algorithm to use in the recipe, can be any of the
        following, `true` (represents Magnitude/Global-Magnitude pruning according to
        global_sparsity), `false` (No pruning), `acdc`, `mfac`, `movement`, `obs` or
        `constant`. Defaults to `true`
    :param quant: `True` if quantization needs to be applied else `False`. Defaults
        to `True`
    :param lr_func: the learning rate schedule function. Defaults to `linear`
    :param recipe_args: additional arguments to pass to recipe_template
    :return: a valid recipe
    """
    return recipe_template(
        model=model,
        pruning=pruning,
        quantization=quant,
        lr=lr_func,
        **recipe_args,
    )


def create_sparse_transfer_recipe(
    model: Optional["Module"] = None,  # noqa: F821
    quant: bool = True,
    lr_func: str = "linear",
    **recipe_args,
) -> str:
    """
    Convenience function to create a sparse transfer recipe

    :param model: an instantiated PyTorch Module, or the local path to a torch.jit
        loadable *.pt file, if supplied then the recipe is built according to this
        architecture
    :param quant: `True` if quantization needs to be applied else `False`. Defaults
        to `True`
    :param lr_func: the learning rate schedule function. Defaults to `linear`
    :param recipe_args: additional arguments to pass to recipe_template
    :return: a valid recipe
    """
    return recipe_template(
        model=model,
        pruning="constant",
        quantization=quant,
        lr=lr_func,
        **recipe_args,
    )


def create_pruning_recipe(
    model: Optional["Module"] = None,  # noqa: F821
    method: str = "true",
    lr_func: str = "linear",
    **recipe_args,
) -> str:
    """
    Convenience function to create a pruning recipe

    :param model: an instantiated PyTorch Module, or the local path to a torch.jit
        loadable *.pt file, if supplied then the recipe is built according to this
        architecture
    :param method: pruning algorithm to use in the recipe, can be any of the
        following, `true` (represents Magnitude/Global-Magnitude pruning according to
        global_sparsity), `false` (No pruning), `acdc`, `mfac`, `movement`, `obs` or
        `constant`. Defaults to `true`
    :param lr_func: the learning rate schedule function. Defaults to `linear`
    :param recipe_args: additional arguments to pass to recipe_template
    :return: a valid recipe
    """

    return recipe_template(
        model=model,
        pruning=method,
        quantization=False,
        lr=lr_func,
        **recipe_args,
    )


def create_quantization_recipe(
    model: Optional["Module"] = None,  # noqa: F821
    method: bool = True,
    lr_func: str = "linear",
    **recipe_args,
) -> str:
    """
    Convenience function to create a quantization recipe

    :param model: an instantiated PyTorch Module, or the local path to a torch.jit
        loadable *.pt file, if supplied then the recipe is built according to this
        architecture
    :param method: `True` if quantization needs to be applied else `False`. Defaults
        to `True`
    :param lr_func: the learning rate schedule function. Defaults to `linear`
    :param recipe_args: additional arguments to pass to recipe_template
    :return: a valid recipe
    """

    return recipe_template(
        model=model,
        pruning=False,
        quantization=method,
        lr=lr_func,
        **recipe_args,
    )
