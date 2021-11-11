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
from typing import Any, Dict, Optional

from sparseml import Framework, execute_in_sparseml_framework
from sparseml.base import detect_frameworks
from sparseml.sparsification.recipe_builder import PruningRecipeBuilder
from sparseml.sparsification.recipe_editor import run_avaialble_recipe_editors


__all__ = ["create_pruning_recipe"]


_LOGGER = logging.getLogger(__name__)


def create_pruning_recipe(
    model: Any,
    save_path: Optional[str] = None,
    analyzer_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    frameworks = detect_frameworks(model)
    framework = Framework.onnx if Framework.onnx in frameworks else frameworks[0]
    if framework is Framework.unknown:
        raise ValueError(f"Unable to detect framework for model {model}")
    _LOGGER.info(f"Creating pruning recipe for model of framework {framework}")

    # Build ModelInfo
    model_info = execute_in_sparseml_framework(model, "ModelInfo", model=model)
    model = model_info.validate_model(model)  # perform any loading/parsing

    # run available analyses
    analyzer_impls = execute_in_sparseml_framework(framework, "get_analyzer_impls")
    analyzer_kwargs = analyzer_kwargs or {}
    if "model" not in analyzer_kwargs:
        analyzer_kwargs["model"] = model
    if "show_progress" not in analyzer_kwargs:
        analyzer_kwargs["show_progress"] = True

    for analyzer_impl in analyzer_impls:
        if not analyzer_impl.available(model_info, **analyzer_kwargs):
            _LOGGER.debug("analyzer %s unavailable", analyzer_impl.__name__)
            continue
        _LOGGER.info(f"Running {analyzer_impl.__name__}")
        analyzer_impl(model_info).run(**analyzer_kwargs)

    # build pruning recipe and run editors
    pruning_recipe = PruningRecipeBuilder(model_info=model_info)
    run_avaialble_recipe_editors(model_info, pruning_recipe)

    if save_path is None:
        return pruning_recipe.build_yaml_str()
    else:
        _LOGGER.info(f"Saving oracle recipe to {save_path}")
        pruning_recipe.save_yaml(save_path)
