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
Classes for creating SparseML recipes through a series of edits based on model
structure and analysis
"""


import logging
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy

from sparseml.sparsification.model_info import (
    LayerInfo,
    ModelInfo,
    PruningSensitivityResult,
)
from sparseml.sparsification.modifier_pruning import GMPruningModifier
from sparseml.sparsification.recipe_builder import (
    PruningRecipeBuilder,
    RecipeYAMLBuilder,
)


__all__ = [
    "RecipeEditor",
    "MobilenetRecipeEditor",
    "SkipFirstLastLayersRecipeEditor",
    "TieredPruningRecipeEditor",
    "run_avaialble_recipe_editors",
]


_LOGGER = logging.getLogger(__name__)


class RecipeEditor(ABC):
    """
    Abstract class for incrementally editing a recipe
    """

    @staticmethod
    @abstractmethod
    def available(model_info: ModelInfo, recipe_builder: RecipeYAMLBuilder) -> bool:
        """
        Abstract method to determine if this RecipeEditor is eligible to edit the
        given recipe

        :param model_info: ModelInfo object of the model the recipe is to be created
            for; should contain layer information and analysis
        :param recipe_builder: RecipeYAMLBuilder of the recipe to be edited
        :return: True if given the inputs, this editor can edit the given recipe. False
            otherwise
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def update_recipe(model_info: ModelInfo, recipe_builder: RecipeYAMLBuilder):
        """
        Abstract method to update a recipe given its current state as a
        RecipeYAMLBuilder and given the analysis in the ModelInfo object

        :param model_info: ModelInfo object of the model the recipe is to be created
            for; should contain layer information and analysis
        :param recipe_builder: RecipeYAMLBuilder of the recipe to update
        """
        raise NotImplementedError()


class MobilenetRecipeEditor(RecipeEditor):
    """
    Recipe Editor for RecipeYAMLBuilder objects with pruning modifiers for mobilenet
    models. Remove mobilenet depthwise layers from their pruning targets.
    Graphs with over 30% depthwise convs and over 30% pointwise convs are assumed to
    be mobilenet(-like) models
    """

    @staticmethod
    def available(model_info: ModelInfo, recipe_builder: RecipeYAMLBuilder) -> bool:
        """
        this editor will be available if the builder has a pruning modifier and
        over 30% of the conv layers in the model are depthwise and 30% are pointwise
        convs

        :param model_info: ModelInfo object of the model the recipe is to be created
            for; should contain layer information and analysis
        :param recipe_builder: RecipeYAMLBuilder of the recipe to be edited
        :return: True if given the inputs, this editor can edit the given recipe. False
            otherwise
        """
        if not recipe_builder.get_modifier_builders(GMPruningModifier):
            return False

        num_convs = 0.0
        num_dw_convs = 0.0
        num_pw_convs = 0.0
        for _, layer_info in model_info.layer_info.items():
            if layer_info.op_type == "conv":
                num_convs += 1.0
                if MobilenetRecipeEditor.is_depthwise_conv(layer_info):
                    num_dw_convs += 1.0
                elif MobilenetRecipeEditor.is_pointwise_conv(layer_info):
                    num_pw_convs += 1.0

        return (
            num_convs
            and num_dw_convs / num_convs >= 0.3
            and (num_pw_convs / num_convs >= 0.3)
        )

    @staticmethod
    def update_recipe(model_info: ModelInfo, recipe_builder: RecipeYAMLBuilder):
        """
        Removes depthwise convs as pruning targets in the recipe

        :param model_info: ModelInfo object of the model the recipe is to be created
            for; should contain layer information and analysis
        :param recipe_builder: RecipeYAMLBuilder of the recipe to update
        """
        for modifier in recipe_builder.get_modifier_builders(GMPruningModifier):
            params = modifier.params
            new_params = []
            for param_name in params:
                param_info = model_info.layer_info[param_name]
                if MobilenetRecipeEditor.is_depthwise_conv(param_info):
                    continue
                new_params.append(param_name)
            modifier.params = new_params

    @staticmethod
    def is_depthwise_conv(layer_info: LayerInfo) -> bool:
        in_channels = layer_info.attributes.get("in_channels")
        groups = layer_info.attributes.get("groups")
        return groups and in_channels and groups == in_channels

    @staticmethod
    def is_pointwise_conv(layer_info: LayerInfo) -> bool:
        kernel_shape = layer_info.attributes.get("kernel_shape")
        groups = layer_info.attributes.get("groups")
        return (
            groups and kernel_shape and tuple(kernel_shape) == (1, 1) and (groups == 1)
        )


class SkipFirstLastLayersRecipeEditor(RecipeEditor):
    """
    Recipe Editor for removing any prunable layers as pruning targets that are either
    the first prunable layer after model input or last prunable layer  before model
    output. The RecipeEditor must have at least one GMPruningModifier to be available

    First and last layers should have the first_prunable_layer and last_prunable_layer
    values set to True in their LayerInfo attributes. if no layers have this attribute
    then the editor will not be available
    """

    @staticmethod
    @abstractmethod
    def available(model_info: ModelInfo, recipe_builder: RecipeYAMLBuilder) -> bool:
        """
        Available if this recipe builder has at least one GMPruningModifier and
        at least one layer info should have the first_prunable_layer or
        last_prunable_layer attribute set to True

        :param model_info: ModelInfo object of the model the recipe is to be created
            for; should contain layer information and analysis
        :param recipe_builder: RecipeYAMLBuilder of the recipe to be edited
        :return: True if given the inputs, this editor can edit the given recipe. False
            otherwise
        """
        if not recipe_builder.get_modifier_builders(GMPruningModifier):
            return False
        for layer_info in model_info.layer_info.values():
            attrs = layer_info.attributes
            if attrs.get("first_prunable_layer") or attrs.get("last_prunable_layer"):
                return True
        return False

    @staticmethod
    @abstractmethod
    def update_recipe(model_info: ModelInfo, recipe_builder: RecipeYAMLBuilder):
        """
        Removes any layers from pruning that are marked as the first or last
        prunable layer(s)

        :param model_info: ModelInfo object of the model the recipe is to be created
            for; should contain layer information and analysis
        :param recipe_builder: RecipeYAMLBuilder of the recipe to update
        """
        for modifier in recipe_builder.get_modifier_builders(GMPruningModifier):
            params = modifier.params
            new_params = []
            for param_name in params:
                param_attrs = model_info.layer_info[param_name].attributes
                if param_attrs.get("first_prunable_layer") or (
                    param_attrs.get("last_prunable_layer")
                ):
                    continue
                new_params.append(param_name)
            modifier.params = new_params


class TieredPruningRecipeEditor(RecipeEditor):
    """
    Recipe Editor for PruningRecipeBuilder objects that have a single pruning modifier.
    Splits the target pruning params into three groups with target sparsities of 1.0,
    0.9, and 0.8 the baseline target sparsity based on their pruning sensitivity
    analysis results

    Baseline target sparsity can be overridden by the base_target_sparsity recipe
    variable.  The mid and low target sparsities as a fraction of the baseline target
    can be overridden with the variables prune_mid_target_pct and prune_low_target_pct
    respectively
    """

    @staticmethod
    def available(model_info: ModelInfo, recipe_builder: RecipeYAMLBuilder) -> bool:
        """
        Determines if this recipe editor can modify the recipe builder. This editor
        is eligible if the recipe builder is a PruningRecipeBuilder with one pruning
        modifier, it is in the pruning_modifiers group, and the ModelInfo includes
        at least one PruningSensitivityResult

        :param model_info: ModelInfo object of the model the recipe is to be created
            for; should contain layer information and analysis
        :param recipe_builder: RecipeYAMLBuilder of the recipe to be edited
        :return: True if given the inputs, this editor can edit the given recipe. False
            otherwise
        """
        return (
            isinstance(recipe_builder, PruningRecipeBuilder)
            and (len(recipe_builder.get_modifier_builders(GMPruningModifier)) == 1)
            and (
                recipe_builder.get_modifier_builders(
                    GMPruningModifier, "pruning_modifiers"
                )
            )
            and any(
                isinstance(result, PruningSensitivityResult)
                for result in model_info.analysis_results
            )
        )

    @staticmethod
    def update_recipe(model_info: ModelInfo, recipe_builder: RecipeYAMLBuilder):
        """
        Abstract method to update a recipe given its current state as a
        RecipeYAMLBuilder and given the analysis in the ModelInfo object

        :param model_info: ModelInfo object of the model the recipe is to be created
            for; should contain layer information and analysis
        :param recipe_builder: RecipeYAMLBuilder of the recipe to update
        """
        # extract pruning modifier
        pruning_modifiers = recipe_builder.get_modifier_group("pruning_modifiers")
        pruning_modifier_idx = [
            idx
            for idx, mod in enumerate(pruning_modifiers)
            if mod.modifier_class is GMPruningModifier
        ][0]
        pruning_modifier = pruning_modifiers.pop(pruning_modifier_idx)
        target_sparsity = recipe_builder.base_target_sparsity

        # get analysis results
        layer_scores = defaultdict(float)
        for result in model_info.analysis_results:
            if not isinstance(result, PruningSensitivityResult):
                continue

            available_sparsities = result.get_available_layer_sparsities()
            if not available_sparsities:
                continue
            analysis_target_sparsity = available_sparsities[-1]
            for sparsity in available_sparsities:
                analysis_target_sparsity = sparsity
                if sparsity >= target_sparsity:
                    break

            for layer_name in pruning_modifier.params:
                layer_scores[layer_name] += result.get_layer_sparsity_score(
                    layer_name, analysis_target_sparsity
                )

        # make cuttoffs, lower sensitivity is better
        scores = numpy.array(list(layer_scores.values()))
        scores_mean = numpy.mean(scores)
        scores_std = numpy.std(scores)
        prune_high_cutoff = scores_mean + (0.25 * scores_std)
        prune_mid_cuttoff = scores_mean + scores_std

        # group params
        params_high = []
        params_mid = []
        params_low = []

        for layer_name, sensitivity in layer_scores.items():
            if sensitivity <= prune_high_cutoff:
                params_high.append(layer_name)
            elif sensitivity <= prune_mid_cuttoff:
                params_mid.append(layer_name)
            else:
                params_low.append(layer_name)

        # set new target sparsities
        recipe_builder.set_variable("prune_mid_target_pct", 0.9)
        recipe_builder.set_variable("prune_low_target_pct", 0.8)
        new_target_sparsities = [
            "eval(base_target_sparsity)",
            "eval(prune_mid_target_pct * base_target_sparsity)",
            "eval(prune_low_target_pct * base_target_sparsity)",
        ]
        new_params = [params_high, params_mid, params_low]

        # create updated modifiers and add all to the modifier group
        for param_group, new_target_sparsity in zip(new_params, new_target_sparsities):
            if not param_group:
                # no params mapped to group
                continue
            updated_modifier = pruning_modifier.copy()
            updated_modifier.params = param_group
            updated_modifier.final_sparsity = new_target_sparsity
            pruning_modifiers.append(updated_modifier)


_EDITORS = [
    SkipFirstLastLayersRecipeEditor,
    MobilenetRecipeEditor,
    TieredPruningRecipeEditor,
]


def run_avaialble_recipe_editors(
    model_info: ModelInfo, recipe_builder: RecipeYAMLBuilder
):
    """
    runs all recipe editors that are available for the given model info and builder

    :param model_info: ModelInfo object of the model the recipe is to be created
        for; should contain layer information and analysis
    :param recipe_builder: RecipeYAMLBuilder of the recipe to update
    """
    editor_names = [editor.__name__ for editor in _EDITORS]
    _LOGGER.debug(
        "checking eligibility and running recipe editors: %s", ", ".join(editor_names)
    )

    for editor_name, editor in zip(editor_names, _EDITORS):
        if not editor.available(model_info, recipe_builder):
            continue
        _LOGGER.info(f"Running recipe editor {editor_name}")
        editor.update_recipe(model_info, recipe_builder)
