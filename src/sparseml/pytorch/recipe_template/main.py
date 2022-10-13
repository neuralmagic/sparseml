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
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.nn import Module

from sparseml.pytorch.sparsification import (
    ACDCPruningModifier,
    ConstantPruningModifier,
    EpochRangeModifier,
    GlobalMagnitudePruningModifier,
    LearningRateFunctionModifier,
    MagnitudePruningModifier,
    MFACPruningModifier,
    MovementPruningModifier,
    OBSPruningModifier,
    QuantizationModifier,
)
from sparseml.pytorch.utils import get_prunable_layers, get_quantizable_layers
from sparseml.sparsification import ModifierYAMLBuilder, RecipeYAMLBuilder


__all__ = [
    "recipe_template",
]


def recipe_template(
    pruning: str = "false",
    quantization: bool = False,
    lr_func: str = "linear",
    mask_type: str = "unstructured",
    global_sparsity: bool = False,
    target: Optional[str] = None,
    model: Union[str, Module, None] = None,
    file_name: str = "recipe.md",
):
    """
    Returns a valid yaml or md recipe based on specified arguments

    :pruning: the pruning algorithm to use in the recipe, can be any of the following,
        `true` (represents Magnitude/Global-Magnitude pruning according to
        global_sparsity), `false` (No pruning), `acdc`, `mfac`, `movement`, `obs` or
        `constant`. Defaults to `false`
    :quantization: True if quantization needs to be applied else False. Defaults to
        False
    :lr_func: the learning rate schedule function. Defaults to `linear`
    :mask_type: the mask_type to use for pruning. Defaults to `unstructured`
    :global_sparsity: if set to True then apply sparsity globally, defaults to
        False
    :target: the target hardware, can be set to `vnni` or `tensorrt`. Defaults to
        None
    :model: an instantiated PyTorch Module, or the local path to a torch.jit loadable
        *.pt file, if supplied then the recipe is built according to this architecture
    :file_name: the filename to save this recipe to. Defaults to `recipe.md`, if the
        file extension is not markdown, then a yaml file is written.
    """

    recipe = _build_recipe(
        global_sparsity=global_sparsity,
        lr_func=lr_func,
        mask_type=mask_type,
        model=model,
        pruning=pruning,
        quantization=quantization,
        target=target,
        convert_to_md=Path(file_name).suffix == ".md",
    )
    _write_recipe_to_file(file_name=file_name, recipe=recipe)
    return recipe


def _build_recipe(
    pruning: str = "false",
    quantization: bool = False,
    lr_func: str = "linear",
    mask_type: str = "unstructured",
    global_sparsity: bool = False,
    target: Optional[str] = None,
    model: Union[str, Module, None] = None,
    convert_to_md: bool = False,
):
    if isinstance(model, str):
        # load model file to in memory Module using torch.jit
        model = torch.jit.load(model)

    mask_type = _validate_mask_type(mask_type=mask_type, target=target)
    recipe_variables = _get_recipe_variables(
        pruning=pruning.lower() != "false",
        quantization=quantization,
        lr_func=lr_func,
        mask_type=mask_type,
        global_sparsity=global_sparsity,
    )

    bool_str_to_pruning_algo: Dict[str, str] = {
        "true": "magnitude",
        "false": "",
    }
    builder_groups = {"training_modifiers": _get_training_modifier_builders()}

    pruning_algo = (
        bool_str_to_pruning_algo[pruning]
        if pruning.lower() in bool_str_to_pruning_algo
        else pruning
    )

    # naive normalization, lower and remove `/` for ac/dc
    pruning_algo = pruning_algo.lower().replace("/", "")

    pruning_algo = "constant" if not pruning_algo and quantization else pruning_algo
    if pruning_algo:
        pruning_builders, pruning_variables = _get_pruning_modifier_builders(
            pruning_algo=pruning_algo, model=model, global_sparsity=global_sparsity
        )
        recipe_variables.update(pruning_variables)
        builder_groups["pruning_modifiers"] = pruning_builders
    if quantization:
        is_post_pruning = pruning_algo not in ["constant", ""]
        quant_builders, quant_variables = _get_quantization_modifier_builders(
            post_pruning=is_post_pruning, target=target
        )
        recipe_variables.update(quant_variables)
        builder_groups["quantization_modifiers"] = quant_builders

    recipe_builder = RecipeYAMLBuilder(
        variables=recipe_variables, modifier_groups=builder_groups
    )
    recipe = recipe_builder.build_yaml_str()

    if convert_to_md:
        recipe = f"---\n{recipe}\n---"
    return recipe


def _get_recipe_variables(
    pruning: bool,
    quantization: bool,
    lr_func: str = "linear",
    mask_type: str = "unstructured",
    global_sparsity: bool = False,
) -> Dict[str, Any]:
    recipe_variables = dict(lr_func=lr_func, init_lr=1.5e-4, final_lr=0)

    if pruning:
        recipe_variables.update(
            dict(
                num_pruning_active_epochs=20,
                num_pruning_finetuning_epochs=10,
                pruning_init_sparsity=0.05,
                pruning_final_sparsity=0.9,
                pruning_update_frequency=0.01,
                mask_type=mask_type,
                global_sparsity=global_sparsity,
                _num_pruning_epochs=(
                    "eval(num_pruning_active_epochs + num_pruning_finetuning_epochs)"
                ),
            )
        )

    if quantization:
        recipe_variables.update(
            dict(
                num_qat_epochs=5,
                num_qat_finetuning_epochs=2.5,
                quantization_submodules="null",
            )
        )
    if pruning and quantization:
        recipe_variables["num_epochs"] = "eval(_num_pruning_epochs + num_qat_epochs)"
    elif pruning and not quantization:
        recipe_variables["num_epochs"] = "eval(_num_pruning_epochs)"
    elif quantization and not pruning:
        recipe_variables["num_epochs"] = "eval(num_qat_epochs)"
    else:
        recipe_variables["num_epochs"] = 35

    return recipe_variables


def _get_training_modifier_builders() -> List[ModifierYAMLBuilder]:
    epoch_modifier = ModifierYAMLBuilder(
        modifier_class=EpochRangeModifier,
        start_epoch=0.0,
        end_epoch="eval(num_epochs)",
    )
    lr_modifier = ModifierYAMLBuilder(
        modifier_class=LearningRateFunctionModifier,
        start_epoch=0.0,
        end_epoch="eval(num_epochs)",
        lr_func="eval(lr_func)",
        init_lr="eval(init_lr)",
        final_lr="eval(final_lr)",
    )
    return [epoch_modifier, lr_modifier]


def _get_pruning_modifier_builders(
    pruning_algo: str,
    model: Optional[Module] = None,
    global_sparsity: bool = False,
) -> Tuple[List[ModifierYAMLBuilder], Dict[str, Any]]:
    pruning_args = dict(
        start_epoch=0.0,
        params="__ALL_PRUNABLE__"
        if model is None
        else _get_prunable_params_for_module(model),
        init_sparsity=0.05,
        final_sparsity=0.8,
        end_epoch=10.0,
        update_frequency=1.0,
        mask_type="eval(mask_type)",
        global_sparsity="eval(global_sparsity)",
        leave_enabled=True,
        inter_func="cubic",
    )
    extra_recipe_variables = {}

    if pruning_algo == "magnitude":
        modifier_class = (
            GlobalMagnitudePruningModifier
            if global_sparsity
            else MagnitudePruningModifier
        )
        # magnitude pruning and global magnitude pruning have
        # their own defaults for global sparsity
        del pruning_args["global_sparsity"]

    elif pruning_algo == "acdc":
        modifier_class = ACDCPruningModifier

        # AC/DC algorithm only specifies a final compression sparsity
        pruning_args["compression_sparsity"] = pruning_args["final_sparsity"]

        del pruning_args["init_sparsity"]
        del pruning_args["final_sparsity"]
        del pruning_args["inter_func"]

    elif pruning_algo == "mfac":
        modifier_class = MFACPruningModifier
        extra_recipe_variables.update(dict(num_grads=64, fisher_block_size=10000))
        pruning_args.update(
            dict(
                num_grads="eval(num_grads)", fisher_block_size="eval(fisher_block_size)"
            )
        )

    elif pruning_algo == "movement":
        modifier_class = MovementPruningModifier
        # movement pruning does not support global_sparsity
        del pruning_args["global_sparsity"]

    elif pruning_algo == "obs":
        modifier_class = OBSPruningModifier
        extra_recipe_variables.update(dict(num_grads=1024, fisher_block_size=50))
        pruning_args.update(
            dict(
                num_grads="eval(num_grads)",
                fisher_block_size="eval(fisher_block_size)",
            )
        )
    elif pruning_algo == "constant":
        modifier_class = ConstantPruningModifier
        pruning_args = dict(start_epoch=0.0, end_epoch=10.0, params="__ALL_PRUNABLE__")
    else:
        raise ValueError(f"Unknown pruning_algo type {pruning_algo}")

    pruning_modifier = ModifierYAMLBuilder(
        modifier_class=modifier_class, **pruning_args
    )
    return [pruning_modifier], extra_recipe_variables


def _get_prunable_params_for_module(model: Module) -> List[str]:
    return [
        f"{name}.weight"
        for name, _ in get_prunable_layers(module=model)
        if "input" not in name and "output" not in name
    ]


def _get_quantizable_params_for_module(model: Module) -> List[str]:
    return [name for name, _ in get_quantizable_layers(module=model)]


def _get_quantization_modifier_builders(
    post_pruning: bool,
    target: str,
    model: Optional[Module] = None,
) -> Tuple[List[ModifierYAMLBuilder], Dict[str, Any]]:
    extra_recipe_variables = {}
    freeze_graph_epoch = (
        "eval(_num_pruning_epochs + num_qat_epochs - num_qat_finetuning_epochs)"
        if post_pruning
        else "eval(num_qat_epochs - num_qat_finetuning_epochs)"
    )
    quantization_modifier = ModifierYAMLBuilder(
        modifier_class=QuantizationModifier,
        start_epoch="eval(_num_pruning_epochs)" if post_pruning else 0.0,
        submodules="eval(quantization_submodules)",
        disable_quantization_observer_epoch=freeze_graph_epoch,
        freeze_bn_stats_epoch=freeze_graph_epoch,
        tensorrt=target == "tensorrt",
        quantize_linear_activations=False,
        quantize_conv_activations=False,
        quantize_embedding_activations=False,
        exclude_module_types=["LayerNorm", "Tanh"],
    )

    extra_recipe_variables["quantization_submodules"] = (
        "null" if not model else _get_quantizable_params_for_module(model=model)
    )
    return [quantization_modifier], extra_recipe_variables


def _validate_mask_type(
    mask_type: str = "unstructured", target: Optional[str] = None
) -> str:
    target_to_mask_type = defaultdict(
        lambda: "unstructured",
        {"vnni": "block4", "tensorrt": "2:4"},
    )

    if mask_type != target_to_mask_type[target]:
        warnings.warn(
            f"The specified mask type {mask_type} and target {target} are "
            f"incompatible, overriding mask_type tp {target_to_mask_type[target]}"
        )
        mask_type = target_to_mask_type[target]
    return mask_type


def _write_recipe_to_file(file_name: str, recipe: str):
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
    with open(file_name, "w") as file:
        file.write(recipe)
