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
from typing import Any, Dict, List, Optional, Union

import torch
from torch.nn import Module

from sparseml.optim import (
    BaseManager,
    add_framework_metadata,
    load_recipe_yaml_str,
    parse_recipe_variables,
    validate_metadata,
)
from sparseml.pytorch.sparsification import Modifier
from sparsezoo.objects import File


__all__ = ["RecipeManagerOneShot"]

_LOGGER = logging.getLogger(__name__)


class RecipeManagerOneShot(BaseManager):
    """
    Recipe manager for handling multiple Modifiers called in a one-shot fashion. Call
    one_shot() to run initialize() for each modifier in recipe.yaml, followed
    by finalize() for each initialized modifier

    Life-cycle:
        - from_yaml(recipe.yaml)
        - one_shot(model, dataloader)
            - initialize
            - finalize
    """

    @staticmethod
    def from_yaml(
        file_path: Union[str, File],
        add_modifiers: Optional[List[Modifier]] = None,
        recipe_variables: Optional[Union[Dict[str, Any], str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Convenience function used to create the manager of multiple modifiers from a
        recipe file.

        :param file_path: the path to the recipe file to load the modifier from, or
            a SparseZoo model stub to load a recipe for a model stored in SparseZoo.
            SparseZoo stubs should be preceded by 'zoo:', and can contain an optional
            '?recipe_type=<type>' parameter. Can also be a SparseZoo File
            object. i.e. '/path/to/local/recipe.md', 'zoo:model/stub/path',
            'zoo:model/stub/path?recipe_type=transfer'. Additionally, a raw
             yaml str is also supported in place of a file path.
        :param add_modifiers: additional modifiers that should be added to the
            returned manager alongside the ones loaded from the recipe file
        :param recipe_variables: additional arguments to override any root variables
            in the recipe with (i.e. num_epochs, init_lr)
        :metadata: additional (to the information provided in the recipe) data to be
            preserved and utilized in the future - for reproducibility and completeness.
        :return: RecipeManagerOneShot() created from the recipe file
        """
        recipe_variables = parse_recipe_variables(recipe_variables)
        yaml_str = load_recipe_yaml_str(file_path, **recipe_variables)
        modifiers = Modifier.load_list(yaml_str)
        if add_modifiers:
            modifiers.extend(add_modifiers)

        validated_metadata = validate_metadata(metadata, yaml_str)

        if metadata is not None:
            validated_metadata = add_framework_metadata(validated_metadata)

        manager = RecipeManagerOneShot(modifiers=modifiers, metadata=validated_metadata)
        return manager

    def one_shot(
        self,
        module: Module,
        data_loader: List,
        device: Optional[str] = "cuda:0",
        initialize_kwargs: Optional[dict] = None,
        finalize_kwargs: Optional[dict] = None,
    ):
        """
        Apply recipe to the model in a one-shot manner, using the provided data loader

        :param model: model to be modified
        :param data_loader: data loader to be used by modifier
        :param device: device to compute on, cpu or cuda:index
        :param initialize_kwargs: Optional kwargs to support specific arguments
            for initializing individual modifiers.
        :param finalize_kwargs: Optional kwargs to support specific arguments
            for finalizing individual modifiers.
        """

        if not torch.cuda.is_available():
            device = "cpu"
            _LOGGER.warning("No GPU available, falling back to CPU")
        module.to(device)

        # used by SparseGPTModifier for OBCQ algorithm
        initialize_kwargs = {"calibration_dataloader": data_loader, "device": device}

        self.initialize(module, **initialize_kwargs)
        self.finalize(module)

    def initialize(
        self,
        module: Module,
        **kwargs,
    ):
        """
        Initializes all modifiers for the given model.

        :param model: the model to modify
        :param kwargs: Optional kwargs to support specific arguments
            for individual modifiers.
        """

        for mod in self.iter_modifiers():
            mod.initialize(module, **kwargs)

    def finalize(self, module: Module = None):
        """
        Handles any finalization of the modifier for the given model.
        Applies any remaining logic and cleans up any hooks or attachments to the model.

        :param model: The model to finalize the modifier for.
            Marked optional so state can still be cleaned up on delete,
            but generally should always be passed in.
        """

        for mod in self.iter_modifiers():
            mod.finalize(module)

    @staticmethod
    def _sort_modifiers_list(modifiers: List[Modifier]) -> List[Modifier]:
        """
        Sort modifiers. For one-shot, the order in the yaml recipe should be respected
        """
        return modifiers
