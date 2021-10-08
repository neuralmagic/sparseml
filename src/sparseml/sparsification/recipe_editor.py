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


from abc import ABC, abstractmethod

from sparseml.sparsification.model_info import ModelInfo
from sparseml.sparsification.recipe_builder import RecipeYAMLBuilder


__all__ = [
    "RecipeEditor",
]


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
