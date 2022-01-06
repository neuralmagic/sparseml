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
Helper variables and functions for integrating SparseML with huggingface/transformers
flows
"""


import os
from typing import Any, Dict

import torch
from transformers.file_utils import WEIGHTS_NAME

from sparseml.pytorch.optim.manager import ScheduledModifierManager


__all__ = [
    "RECIPE_NAME",
    "preprocess_state_dict",
    "load_recipe",
]


RECIPE_NAME = "recipe.yaml"


def load_recipe(pretrained_model_name_or_path: str) -> str:
    """
    Get path to recipe from the model directory

    :param pretrained_model_name_or_path: path to model directory
    :return: path to recipe
    """
    recipe = None
    if pretrained_model_name_or_path is not None:
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            if os.path.isfile(os.path.join(pretrained_model_name_or_path, RECIPE_NAME)):
                recipe = os.path.join(pretrained_model_name_or_path, RECIPE_NAME)
    return recipe


def preprocess_state_dict(pretrained_model_name_or_path: str) -> Dict[str, Any]:
    """
    Restore original parameter names that were changed by QAT process

    :param pretrained_model_name_or_path: name or path to model
    """
    state_dict = None
    if pretrained_model_name_or_path is not None:
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            if os.path.isfile(os.path.join(pretrained_model_name_or_path, RECIPE_NAME)):
                recipe = os.path.join(pretrained_model_name_or_path, RECIPE_NAME)
                manager = ScheduledModifierManager.from_yaml(recipe)
                modifiers = [m.__class__.__name__ for m in manager.modifiers]
                is_qat_recipe = "QuantizationModifier" in modifiers
            else:
                is_qat_recipe = False
            if os.path.isfile(
                os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
            ):
                archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                state_dict = torch.load(archive_file, map_location="cpu")
                removed_keys = (
                    [
                        key
                        for key in state_dict
                        if (
                            key.endswith(".module.weight")
                            or key.endswith(".module.bias")
                        )
                    ]
                    if is_qat_recipe
                    else []
                )
                for key in removed_keys:
                    new_key = key.replace(".module", "")
                    state_dict[new_key] = state_dict[key]
                    state_dict.pop(key)
    return state_dict
