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

from contextlib import suppress
from pathlib import Path

from huggingface_hub import hf_hub_download
from sparsezoo import Model


__all__ = ["fetch_recipe_path"]


def fetch_recipe_path(target: str):
    """
    Fetches the recipe path for the given target.
    This method will also download the recipe if it is not
    already downloaded.

    Takes care of three scenarios:
    1. target is a local path to a model directory
        (looks for recipe.yaml in the directory)
    2. target is a SparseZoo stub (downloads and
        returns the path to the default recipe)
    3. target is a HuggingFace stub (downloads and
        returns the path to the default recipe)

    :param target: The target to fetch the recipe path for
        can be a local path, SparseZoo stub, or HuggingFace stub
    :return: The path to the recipe for the target
    """
    DEFAULT_RECIPE_NAME = "recipe.yaml"
    if Path(target).exists():
        # target is a local path
        potential_recipe_path = Path(target) / DEFAULT_RECIPE_NAME
        return str(potential_recipe_path) if potential_recipe_path.exists() else None

    # Recipe must be downloaded

    recipe_path = None
    if target.startswith("zoo:"):
        # target is a SparseZoo stub
        sparsezoo_model = Model(source=target)
        with suppress(Exception):
            # suppress any errors if the recipe is not found on SparseZoo
            recipe_path = sparsezoo_model.recipes.default().download()
        return recipe_path

    # target is a HuggingFace stub
    with suppress(Exception):
        # suppress any errors if the recipe is not found on HuggingFace
        recipe_path = hf_hub_download(repo_id=target, filename=DEFAULT_RECIPE_NAME)

    return recipe_path
