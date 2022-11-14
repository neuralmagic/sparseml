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

__all__ = [
    "create_recipe",
    "create_sparse_transfer_recipe",
    "create_pruning_recipe",
    "create_quantization_recipe",
]


def create_recipe(model=None, **recipe_args):
    """
    Convenience function to create a recipe based on supplied args and kwargs
    """


def create_sparse_transfer_recipe(
    model=None, quant=True, lr_func="linear", **recipe_args
):
    """
    Convenience function to create a sparse transfer recipe
    """


def create_pruning_recipe(
    model=None, quantization=False, lr_func="linear", **recipe_args
):
    """
    Convenience function to create a pruning recipe
    """


def create_quantization_recipe(
    model=None, pruning=False, lr_func="linear", **recipe_args
):
    """
    Convenience function to create a quantization
    """
