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
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.nn import Module

from sparseml.pytorch.recipe_template.description import DESCRIPTION
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
from sparseml.pytorch.utils import get_prunable_layers
from sparseml.sparsification import ModifierYAMLBuilder, RecipeYAMLBuilder


__all__ = [
    "recipe_template",
]


def recipe_template(
    pruning: Optional[str] = None,
    quantization: bool = False,
    lr_func: str = "linear",
    mask_type: str = "unstructured",
    global_sparsity: bool = False,
    target: Optional[str] = None,
    model: Union[str, Module, None] = None,
    file_name: Optional[str] = None,
) -> str:
    """
    Returns a valid yaml or md recipe based on specified arguments

    :pruning: optional pruning algorithm to use in the recipe, can be any of the
    following,
        `true` (represents Magnitude/Global-Magnitude pruning according to
        global_sparsity), `false` (No pruning), `acdc`, `mfac`, `movement`, `obs` or
        `constant`. Defaults to None
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
    :file_name: an optional filename to save this recipe to. If specified the extension
        is used to determine if file should be written in markdown or yaml syntax. If
        not specified recipe is not written to a file
    """

    if isinstance(model, str):
        # load model file to in memory Module using torch.jit
        model = torch.jit.load(model)

    mask_type: str = _validate_mask_type(mask_type=mask_type, target=target)
    pruning: str = _validate_pruning(pruning=pruning, quantization=quantization)
    recipe: str = _build_recipe_template(
        pruning=pruning,
        quantization=quantization,
        lr_func=lr_func,
        mask_type=mask_type,
        global_sparsity=global_sparsity,
        target=target,
        model=model,
    )

    if file_name is not None:
        if Path(file_name).suffix == ".md":
            recipe = _add_description(recipe=recipe)

        _write_recipe_to_file(file_name=file_name, recipe=recipe)
    return recipe


def _validate_pruning(pruning: Optional[str] = None, quantization: bool = False) -> str:
    # normalize pruning algo name
    pruning = (pruning or "").lower().replace("/", "")

    if pruning == "true":
        pruning = "magnitude"
    elif pruning == "false":
        pruning = "constant" if quantization else ""
    return pruning


def _validate_mask_type(
    mask_type: str = "unstructured", target: Optional[str] = None
) -> str:
    target_to_mask_type = defaultdict(
        lambda: "unstructured",
        {"vnni": "block4", "tensorrt": "2:4"},
    )

    if target is not None and mask_type != target_to_mask_type[target]:
        raise ValueError(
            f"The specified mask type {mask_type} and target {target} are "
            f"incompatible, try overriding mask_type to {target_to_mask_type[target]}"
        )
    return mask_type


def _build_recipe_template(
    pruning: str,
    quantization: bool,
    lr_func: str,
    mask_type: str,
    global_sparsity: bool = False,
    target: Optional[str] = None,
    model: Union[Module, None] = None,
) -> str:
    pruning_was_applied: bool = pruning not in ["constant", ""]
    recipe_variables: Dict[str, Any] = _get_base_recipe_variables(
        pruning=pruning_was_applied,
        quantization=quantization,
        lr_func=lr_func,
        mask_type=mask_type,
        global_sparsity=global_sparsity,
    )

    builder_groups = {"training_modifiers": _get_training_builders()}

    if pruning:
        pruning_builders, pruning_variables = _get_pruning_builders_and_variables(
            pruning_algo=pruning, model=model, global_sparsity=global_sparsity
        )
        recipe_variables.update(pruning_variables)
        builder_groups["pruning_modifiers"] = pruning_builders

    if quantization:
        quant_builders, quant_variables = _get_quantization_builders_and_variables(
            post_pruning=pruning_was_applied, target=target
        )
        recipe_variables.update(quant_variables)
        builder_groups["quantization_modifiers"] = quant_builders

    recipe_builder = RecipeYAMLBuilder(
        variables=recipe_variables, modifier_groups=builder_groups
    )
    recipe = recipe_builder.build_yaml_str()
    return recipe


def _add_description(recipe: str, description: str = DESCRIPTION) -> str:
    return f"---\n{recipe}\n---{description}"


def _write_recipe_to_file(file_name: str, recipe: str):
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
    with open(file_name, "w") as file:
        file.write(recipe)


def _get_base_recipe_variables(
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
    recipe_variables["num_epochs"] = _get_num_epochs(
        pruning=pruning,
        quantization=quantization,
    )

    return recipe_variables


def _get_num_epochs(pruning: bool, quantization: bool) -> Union[int, str]:
    if pruning and quantization:
        return "eval(_num_pruning_epochs + num_qat_epochs)"
    if pruning and not quantization:
        return "eval(_num_pruning_epochs)"
    if quantization and not pruning:
        return "eval(num_qat_epochs)"
    # default placeholder epoch number if no
    # pruning or quantization present
    return 35


def _get_training_builders() -> List[ModifierYAMLBuilder]:
    epoch_modifier_builder = ModifierYAMLBuilder(
        modifier_class=EpochRangeModifier,
        start_epoch=0.0,
        end_epoch="eval(num_epochs)",
    )
    lr_modifier_builder = ModifierYAMLBuilder(
        modifier_class=LearningRateFunctionModifier,
        start_epoch=0.0,
        end_epoch="eval(num_epochs)",
        lr_func="eval(lr_func)",
        init_lr="eval(init_lr)",
        final_lr="eval(final_lr)",
    )
    return [epoch_modifier_builder, lr_modifier_builder]


def _get_pruning_builders_and_variables(
    pruning_algo: str,
    model: Optional[Module] = None,
    global_sparsity: bool = False,
) -> Tuple[List[ModifierYAMLBuilder], Dict[str, Any]]:
    prunable_params = (
        "__ALL_PRUNABLE__"
        if model is None
        else _get_prunable_params_for_module(model=model)
    )

    pruning_arguments = dict(
        start_epoch=0.0,
        params=prunable_params,
        init_sparsity=0.05,
        final_sparsity=0.8,
        end_epoch=10.0,
        update_frequency=1.0,
        mask_type="eval(mask_type)",
        global_sparsity="eval(global_sparsity)",
        leave_enabled=True,
        inter_func="cubic",
    )

    pruning_recipe_variables = {}

    if pruning_algo == "magnitude":
        modifier_class = (
            GlobalMagnitudePruningModifier
            if global_sparsity
            else MagnitudePruningModifier
        )
        # magnitude pruning and global magnitude pruning have
        # their own defaults for global sparsity
        del pruning_arguments["global_sparsity"]

    elif pruning_algo == "acdc":
        modifier_class = ACDCPruningModifier

        # AC/DC algorithm only specifies a final compression sparsity
        pruning_arguments["compression_sparsity"] = pruning_arguments["final_sparsity"]

        # delete other sparsity arguments
        del pruning_arguments["init_sparsity"]
        del pruning_arguments["final_sparsity"]
        del pruning_arguments["inter_func"]

    elif pruning_algo == "mfac":
        modifier_class = MFACPruningModifier
        pruning_recipe_variables.update(dict(num_grads=64, fisher_block_size=10000))
        pruning_arguments.update(
            dict(
                num_grads="eval(num_grads)",
                fisher_block_size="eval(fisher_block_size)",
            )
        )

    elif pruning_algo == "movement":
        modifier_class = MovementPruningModifier

        # movement pruning does not support global_sparsity
        if global_sparsity:
            raise ValueError("Movement pruning does not support `global_sparsity`")

        del pruning_arguments["global_sparsity"]

    elif pruning_algo == "obs":
        modifier_class = OBSPruningModifier
        pruning_recipe_variables.update(dict(num_grads=1024, fisher_block_size=50))
        pruning_arguments.update(
            dict(
                num_grads="eval(num_grads)",
                fisher_block_size="eval(fisher_block_size)",
            )
        )
    elif pruning_algo == "constant":
        modifier_class = ConstantPruningModifier

        # constant pruning modifier only specifies start_epoch, end_epoch and
        # params
        pruning_arguments = dict(
            start_epoch=0.0,
            params="__ALL_PRUNABLE__",  # preferred for constant pruning
        )
    else:
        raise ValueError(f"Unknown pruning_algo type {pruning_algo}")

    pruning_modifier_builder = ModifierYAMLBuilder(
        modifier_class=modifier_class, **pruning_arguments
    )
    return [pruning_modifier_builder], pruning_recipe_variables


def _get_prunable_params_for_module(model: Module) -> List[str]:
    return [
        f"{name}.weight"
        for name, _ in get_prunable_layers(module=model)
        if "input" not in name and "output" not in name
    ]


def _get_quantization_builders_and_variables(
    post_pruning: bool,
    target: str,
) -> Tuple[List[ModifierYAMLBuilder], Dict[str, Any]]:
    quant_recipe_variables = {}
    freeze_graph_epoch = (
        "eval(_num_pruning_epochs + num_qat_epochs - num_qat_finetuning_epochs)"
        if post_pruning
        else "eval(num_qat_epochs - num_qat_finetuning_epochs)"
    )
    quant_modifier_builder = ModifierYAMLBuilder(
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

    quant_recipe_variables["quantization_submodules"] = "null"

    return [quant_modifier_builder], quant_recipe_variables
