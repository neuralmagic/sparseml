#!/usr/bin/env python

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
Usage: sparseml.recipe_template [OPTIONS]

  Utility to create a recipe template based on specified options

  Example for using sparseml.recipe_template:

       `sparseml.recipe_template --pruning true --quantization true`

       `sparseml.recipe_template --quantization true --target vnni --lr constant `

Options:
  --version                       Show the version and exit.  [default: False]
  --pruning [true|false|gmp|acdc|mfac|movement|constant]
                                  Specify if recipe should include pruning
                                  steps, can also take in the name of a
                                  pruning algorithm  [default: false]
  --quantization [true|false]     Specify if recipe should include
                                  quantization steps, can also take in the
                                  name of the target hardware  [default:
                                  false]
  --lr, --learning_rate [constant|cyclic|stepped|exponential|linear]
                                  Specify learning rate schedule function
                                  [default: constant]
  --target [vnni|tensorrt|default]
                                  Specify target hardware type for current
                                  recipe  [default: default]
  --distillation                  Add distillation support to the recipe
                                  [default: False]
  --file_name TEXT                The file name to output recipe to  [default:
                                  recipe.md]
  --help                          Show this message and exit.  [default:
                                  False]
"""

import logging

import click
from sparseml.pytorch import recipe_template
from sparseml.version import __version__


_LOGGER = logging.getLogger(__name__)
PRUNING_ALGOS = ["true", "false", "gmp", "acdc", "mfac", "movement", "constant"]


@click.command(context_settings=(dict(show_default=True)))
@click.version_option(version=__version__)
@click.option(
    "--pruning",
    type=click.Choice(PRUNING_ALGOS, case_sensitive=False),
    default="false",
    help="Specify if recipe should include pruning steps, can also take in the "
    "name of a pruning algorithm",
)
@click.option(
    "--quantization",
    type=click.Choice(["true", "false"], case_sensitive=False),
    default="false",
    help="Specify if recipe should include quantization steps, can also take in "
    "the name of the target hardware",
)
@click.option(
    "--lr",
    "--learning_rate",
    type=click.Choice(
        ["constant", "cyclic", "stepped", "exponential", "linear"], case_sensitive=False
    ),
    default="constant",
    help="Specify learning rate schedule function",
)
@click.option(
    "--target",
    type=click.Choice(["vnni", "tensorrt", "default"], case_sensitive=False),
    default="default",
    help="Specify target hardware type for current recipe",
)
@click.option(
    "--distillation",
    is_flag=True,
    help="Add distillation support to the recipe",
)
@click.option(
    "--file_name",
    type=str,
    default="recipe.md",
    help="The file name to output recipe to",
)
def main(**kwargs):
    """
    Utility to create a recipe template based on specified options

    Example for using sparseml.pytorch.recipe_template:

         `sparseml.recipe_template --pruning true --quantization true`

         `sparseml.recipe_template --quantization true --target vnni --lr constant `
    """
    _LOGGER.debug(f"{kwargs}")
    if kwargs.get("quantization") == "true":
        target = kwargs.get("target")
        # override mask type based on target
        if target == "vnni":
            kwargs["mask_type"] = "block4"
        elif target == "tensorrt":
            kwargs["mask_type"] = "2:4"
    template = recipe_template(**kwargs)
    print(f"Template:\n{template}")


if __name__ == "__main__":
    main()
