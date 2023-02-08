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
Contains base code related to modifier managers: modifier managers handle
grouping modifiers and running them together.
Also handles loading modifiers from yaml files
"""


from typing import Any, Dict, List, Optional, Union

from tensorflow import Tensor

from sparseml.keras.optim.modifier import Modifier, ScheduledModifier
from sparseml.keras.utils.compat import keras
from sparseml.keras.utils.logger import KerasLogger
from sparseml.optim import (
    BaseManager,
    add_framework_metadata,
    load_recipe_yaml_str,
    parse_recipe_variables,
    validate_metadata,
)
from sparsezoo.objects import File


__all__ = ["ScheduledModifierManager"]


class ScheduledModifierManager(BaseManager, Modifier):
    """
    The base modifier manager, handles managing multiple ScheduledModifier.
    """

    @staticmethod
    def from_yaml(
        file_path: Union[str, File],
        add_modifiers: List[Modifier] = None,
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
            'zoo:model/stub/path?recipe_type=transfer'
        :param add_modifiers: additional modifiers that should be added to the
            returned manager alongside the ones loaded from the recipe file
        :param recipe_variables: additional variable values to override the recipe
            with (i.e. num_epochs, init_lr)
        :metadata: additional (to the information provided in the recipe) data to be
            preserved and utilized in the future - for reproducibility and completeness.
        :return: ScheduledModifierManager() created from the recipe file
        """
        recipe_variables = parse_recipe_variables(recipe_variables)
        yaml_str = load_recipe_yaml_str(file_path, **recipe_variables)
        modifiers = Modifier.load_list(yaml_str)
        if add_modifiers:
            modifiers.extend(add_modifiers)

        validated_metadata = validate_metadata(metadata, yaml_str)

        if metadata is not None:
            validated_metadata = add_framework_metadata(
                validated_metadata, keras_version=keras.__version__
            )

        manager = ScheduledModifierManager(
            modifiers=modifiers, metadata=validated_metadata
        )
        return manager

    def __init__(
        self,
        modifiers: List[ScheduledModifier],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(modifiers=modifiers, metadata=metadata)
        self._optimizer = None

    def modify(
        self,
        model: Union[keras.Model, keras.Sequential],
        optimizer: keras.optimizers.Optimizer,
        steps_per_epoch: int,
        loggers: Union[KerasLogger, List[KerasLogger]] = None,
        input_tensors: Tensor = None,
    ):
        """
        Modify the model and optimizer based on the requirements of modifiers

        :param model: model to modify
        :param optimizer: optimizer to modify
        :param steps_per_epoch: number of steps per epoch
        :param loggers: list of loggers
        :param input_tensors: optional input tensor
        :return: model, optimizer, callbacks
        """

        # Different modifiers might have logging callbacks a same global variables,
        # thus modifiers need to be sorted increasing based on their start steps to
        # make sure logging on shared variables reflect the latest effect
        self._modifiers.sort(key=lambda mod: mod.start_epoch)

        callbacks = []
        for mod in self._modifiers:
            model, optimizer, callback = mod.modify(
                model,
                optimizer,
                steps_per_epoch,
                loggers=loggers,
                input_tensors=input_tensors,
            )
            if callback is None:
                continue
            if isinstance(callback, list):
                callbacks = callbacks + callback
            elif isinstance(callback, keras.callbacks.Callback):
                callbacks.append(callback)
            else:
                raise RuntimeError("Invalid callback type")
        self._optimizer = optimizer
        return model, optimizer, callbacks

    def finalize(self, model: keras.Model):
        """
        Remove extra information related to the modifier from the model that is
        not necessary for exporting

        :param model: a Keras model
        :return: a new Keras model
        """
        for mod in self._modifiers:
            model = mod.finalize(model)
        return model
