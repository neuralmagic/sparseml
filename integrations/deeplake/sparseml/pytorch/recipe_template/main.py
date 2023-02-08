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
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.nn import Module

from sparseml.pytorch.recipe_template.description import DESCRIPTION
from sparseml.pytorch.sparsification import (
    ACDCPruningModifier,
    ConstantPruningModifier,
    DistillationModifier,
    EpochRangeModifier,
    GlobalMagnitudePruningModifier,
    LearningRateFunctionModifier,
    MagnitudePruningModifier,
    MFACPruningModifier,
    MovementPruningModifier,
    OBSPruningModifier,
)
from sparseml.pytorch.sparsification.quantization.legacy_modifier_quantization import (
    QuantizationModifier,
)
from sparseml.pytorch.utils import get_prunable_layers
from sparseml.sparsification import ModifierYAMLBuilder, RecipeYAMLBuilder


__all__ = [
    "recipe_template",
]

_LOGGER = logging.getLogger(__name__)


def recipe_template(
    pruning: Union[str, bool, None] = None,
    quantization: Union[bool, str] = False,
    lr: str = "linear",
    mask_type: str = "unstructured",
    global_sparsity: bool = True,
    target: Optional[str] = None,
    model: Union[str, Module, None] = None,
    file_name: Optional[str] = None,
    num_epochs: float = 20.0,
    init_lr: float = 0.001,
    final_lr: float = 0.0,
    sparsity: float = 0.8,
    distillation: bool = False,
    hardness: float = 0.5,
    temperature: float = 2.0,
) -> str:
    """
    Returns a valid yaml or md recipe based on specified arguments

    :param pruning: optional pruning algorithm to use in the recipe, can be any of the
        following,`true` (represents Magnitude/Global-Magnitude pruning according to
        global_sparsity), `false` (No pruning), `acdc`, `mfac`, `movement`, `obs` or
        `constant`. Can also be a bool. Defaults to None
    :param quantization: True if quantization needs to be applied else False. Defaults
        to False. Can also be string representation of boolean values i.e `true` or
        `false`
    :param lr: the learning rate schedule function. Defaults to `linear`
    :param mask_type: the mask_type to use for pruning. Defaults to `unstructured`
    :param global_sparsity: if set to True then apply sparsity globally, defaults to
        False
    :param target: the target hardware, can be set to `vnni` or `tensorrt`. Defaults to
        None
    :param model: an instantiated PyTorch Module, or the local path to a torch.jit
        loadable *.pt file, if supplied then the recipe is built according to this
        architecture
    :param file_name: an optional filename to save this recipe to. If specified the
        extension is used to determine if file should be written in markdown
        or yaml syntax. If not specified recipe is not written to a file
    :param num_epochs: total number of epochs to target in recipe, default 20
    :param init_lr: target initial learning rate, default 0.001
    :param final_lr: target final learning rate, default 0.0
    :param sparsity: target model sparsity, default 0.8
    :param distillation: add distillation support to the recipe. default is
        `False`
    :param hardness: [only used if distillation is set] how much to weight the
        distillation loss vs the base loss (e.g. hardness of 0.6 will return
        0.6 * distill_loss + 0.4 * base_loss). default is 0.5
    :param temperature: [only used if distillation is set] temperature applied
        to teacher and student softmax for distillation. default is 2.0
    :return: A valid string recipe based on the arguments
    """

    if isinstance(model, str):
        # load model file to in memory Module using torch.jit
        model = torch.jit.load(model)

    quantization: bool = _validate_quantization(quantization=quantization)
    mask_type: str = _validate_mask_type(mask_type=mask_type, target=target)
    pruning: str = _validate_pruning(pruning=pruning, quantization=quantization)
    recipe: str = _build_recipe_template(
        pruning=pruning,
        quantization=quantization,
        lr_func=lr,
        mask_type=mask_type,
        global_sparsity=global_sparsity,
        target=target,
        model=model,
        num_epochs=num_epochs,
        init_lr=init_lr,
        final_lr=final_lr,
        sparsity=sparsity,
        distillation=distillation,
        hardness=hardness,
        temperature=temperature,
    )

    if file_name is not None:
        if Path(file_name).suffix == ".md":
            recipe = _add_description(recipe=recipe)

        _write_recipe_to_file(file_name=file_name, recipe=recipe)
    return recipe


def _validate_quantization(quantization: Union[bool, str]) -> bool:
    if isinstance(quantization, str):
        quantization = quantization.lower() == "true"
    if not isinstance(quantization, bool):
        raise ValueError("`quantization` must be a bool")
    return quantization


def _validate_pruning(
    pruning: Union[str, bool, None] = None, quantization: bool = False
) -> str:
    # normalize pruning algo name
    if isinstance(pruning, bool):
        pruning = str(pruning)

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
    global_sparsity: bool = True,
    target: Optional[str] = None,
    model: Union[Module, None] = None,
    num_epochs: float = 20.0,
    init_lr: float = 0.001,
    final_lr: float = 0.0,
    sparsity: float = 0.8,
    distillation: bool = False,
    hardness: float = 0.5,
    temperature: float = 2.0,
) -> str:
    pruning_was_applied: bool = pruning not in ["constant", ""]
    recipe_variables: Dict[str, Any] = _get_base_recipe_variables(
        pruning=pruning_was_applied,
        quantization=quantization,
        lr_func=lr_func,
        mask_type=mask_type,
        global_sparsity=global_sparsity,
        num_epochs=num_epochs,
        init_lr=init_lr,
        final_lr=final_lr,
        sparsity=sparsity,
        distillation=distillation,
        hardness=hardness,
        temperature=temperature,
    )

    builder_groups = {"training_modifiers": _get_training_builders()}

    if pruning:
        pruning_builders, pruning_variables = _get_pruning_builders_and_variables(
            pruning_algo=pruning,
            model=model,
            global_sparsity=global_sparsity,
        )
        recipe_variables.update(pruning_variables)
        builder_groups["pruning_modifiers"] = pruning_builders

    if quantization:
        quant_builders, quant_variables = _get_quantization_builders_and_variables(
            post_pruning=pruning_was_applied, target=target
        )
        recipe_variables.update(quant_variables)
        builder_groups["quantization_modifiers"] = quant_builders

    if distillation:
        builder_groups["distillation_modifiers"] = _get_distillation_builders()

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
    _LOGGER.info(f"Recipe written to file {file_name}")


def _get_base_recipe_variables(
    pruning: bool,
    quantization: bool,
    lr_func: str = "linear",
    mask_type: str = "unstructured",
    global_sparsity: bool = True,
    num_epochs: float = 20.0,
    init_lr: float = 0.001,
    final_lr: float = 0.0,
    sparsity: float = 0.8,
    distillation: bool = False,
    hardness: float = 0.5,
    temperature: float = 2.0,
) -> Dict[str, Any]:

    recipe_variables = dict(
        lr_func=lr_func, init_lr=init_lr, final_lr=final_lr, num_epochs=num_epochs
    )

    num_qat_epochs = 0
    if quantization:
        num_qat_epochs = 5.0 if num_epochs >= 15.0 else 2.0
        recipe_variables.update(
            dict(
                num_qat_epochs=num_qat_epochs,
                num_qat_finetuning_epochs=num_qat_epochs / 2.0,
                quantization_submodules="null",
            )
        )

    if pruning:
        num_pruning_active_epochs = (num_epochs - num_qat_epochs) / 2.0
        recipe_variables.update(
            dict(
                num_pruning_active_epochs=num_pruning_active_epochs,
                num_pruning_finetuning_epochs=0.5 * (num_epochs - num_qat_epochs),
                pruning_init_sparsity=min(0.05, sparsity),  # enforce init <= final
                pruning_final_sparsity=sparsity,
                pruning_update_frequency=(
                    1.0
                    if num_pruning_active_epochs > 20
                    else num_pruning_active_epochs / 20.0
                ),
                mask_type=mask_type,
                global_sparsity=global_sparsity,
                _num_pruning_epochs=(
                    "eval(num_pruning_active_epochs + num_pruning_finetuning_epochs)"
                ),
            )
        )

    if distillation:
        recipe_variables.update(
            dict(
                distillation_hardness=hardness,
                distillation_temperature=temperature,
            )
        )

    return recipe_variables


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
    global_sparsity: bool = True,
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
        end_epoch="eval(num_pruning_active_epochs)",
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

        # constant pruning modifier only specifies start_epoch and params
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


def _get_distillation_builders():
    distillation_modifier_builder = ModifierYAMLBuilder(
        modifier_class=DistillationModifier,
        start_epoch=0.0,
        end_epoch="eval(num_epochs)",
        hardness="eval(distillation_hardness)",
        temperature="eval(distillation_temperature)",
    )
    return [distillation_modifier_builder]
