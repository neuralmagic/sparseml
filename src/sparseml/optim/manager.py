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
Code related to managers that is shared across frameworks.
Managers control groups of modifiers to allow modifying the training process of a model;
ex to perform model pruning.
"""

import math
from collections import OrderedDict
from copy import deepcopy
from functools import cmp_to_key
from typing import Dict, Generator, List, Union

from sparseml.optim.modifier import BaseModifier, BaseObject, ModifierProp
from sparseml.sparsification.types import SparsificationTypes
from sparseml.utils import clean_path, create_parent_dirs


__all__ = ["BaseManager"]


class BaseManager(BaseObject):
    """
    Parent class meant to be used for all managers.
    Handles base implementations for properties and methods.

    :param modifiers: the modifiers to wrap
    """

    def __init__(
        self,
        modifiers: Union[List[BaseModifier], Dict[str, List[BaseModifier]]],
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(modifiers, List):
            # sort modifiers by when they start and end so that later modifiers
            # can overwrite in a deterministic order such as when initializing
            self._modifiers = _sort_modifiers_list(modifiers)
        elif isinstance(modifiers, Dict):
            # staged recipe
            # sort modifiers of each stage by start/end as above then sort stages
            # by their modifiers
            modifiers = {
                stage: _sort_modifiers_list(stage_modifiers)
                for stage, stage_modifiers in modifiers.items()
            }
            self._modifiers = OrderedDict(
                sorted(
                    modifiers.items(),
                    key=cmp_to_key(
                        lambda item_1, item_2: BaseModifier.comparator_lists(
                            item_1[1], item_2[1]
                        )
                    ),
                )
            )

        else:
            raise ValueError(
                "modifiers type must be List[BaseModifier] or "
                f"Dict[str, List[BaseModifier]] found {type(modifiers)}"
            )

    def __del__(self):
        for mod in self.iter_modifiers():
            del mod

        self._modifiers.clear()

    def __str__(self) -> str:
        return "\n".join(self.to_string_lines())

    def __eq__(self, compare: object) -> bool:
        return str(self) == str(compare)

    @classmethod
    def compose_staged(
        cls,
        base_recipe: Union[str, "BaseManager"],
        additional_recipe: Union[str, "BaseManager"],
        keep_original_epochs: bool = False,
    ) -> "BaseManager":
        """
        composes two recipes into a multi-stage recipe where epochs
        for additional_recipe are overwritten to come after base_recipe

        :param base_recipe: base recipe to compose multi stage recipe with.
            May be a string YAML recipe, file path, or Manager object
        :param additional_recipe: additional recipe whose stages will be added
            to the base recipe. epoch ranges for additional_recipe will be adjusted
            to come after base_recipe unless keep_original_epochs is set.
            May be a string YAML recipe, file path, or Manager object
        :param keep_original_epochs: by default, epochs in additional_recipe will
            be overwritten to come after base_recipe. setting keep_original_epochs
            to True prevents this behavior. Default is False
        :return: framework Manager object with the loaded composed recipe
        """

        # will load using class implementation of from_yaml
        # will fail from BaseModifier
        if not isinstance(base_recipe, BaseManager):
            base_recipe = cls.from_yaml(base_recipe)
        if not isinstance(additional_recipe, BaseManager):
            additional_recipe = cls.from_yaml(additional_recipe)

        if isinstance(base_recipe.modifiers, OrderedDict):
            raise ValueError(
                "non-staged recipes not yet supported for Manager.compose_staged "
                "found base_recipe with non_staged modifiers"
            )
        if isinstance(additional_recipe.modifiers, OrderedDict):
            raise ValueError(
                "non-staged recipes not yet supported for Manager.compose_staged "
                "found additional_recipe with non_staged modifiers"
            )

        base_stages = deepcopy(base_recipe.modifiers)
        additional_stages = deepcopy(additional_recipe.modifiers)

        base_keys = set(base_stages.keys())
        additional_keys = set(additional_stages.keys())
        keys_intersection = base_keys.intersection(additional_keys)
        if keys_intersection:
            raise ValueError(
                "base and additional recipe must not share any stage names. "
                f"found overlapping stage names: {list(keys_intersection)}"
            )

        if not keep_original_epochs:
            # update additional modifier epochs
            base_end_epoch = base_recipe.max_epochs
            for additional_modifiers in additional_stages.values():
                for additional_modifier in additional_modifiers:
                    if hasattr(additional_modifier, "start_epoch"):
                        additional_modifier.start_epoch += base_end_epoch
                    if hasattr(additional_modifier, "end_epoch"):
                        additional_modifier.end_epoch += base_end_epoch

        combined_stages = base_stages
        combined_stages.update(additional_stages)
        return cls(combined_stages)

    @ModifierProp(serializable=False)
    def modifiers(self) -> Union[List[BaseModifier], Dict[str, List[BaseModifier]]]:
        """
        :return: list of all SparseML modifiers in the managed recipe or dictionary
            of modifier stages to list of those modifiers
        """
        return self._modifiers

    @ModifierProp(serializable=False)
    def epoch_modifiers(self) -> List[BaseModifier]:
        """
        :return: list of all SparseML modifiers in the managed recipe that modify the
            epoch range
        """
        return [
            mod
            for mod in self.iter_modifiers()
            if SparsificationTypes.epoch in mod.sparsification_types
        ]

    @ModifierProp(serializable=False)
    def learning_rate_modifiers(self) -> List[BaseModifier]:
        """
        :return: list of all SparseML modifiers in the managed recipe that modify the
            LearningRate schedule
        """
        return [
            mod
            for mod in self.iter_modifiers()
            if SparsificationTypes.learning_rate in mod.sparsification_types
        ]

    @ModifierProp(serializable=False)
    def pruning_modifiers(self) -> List[BaseModifier]:
        """
        :return: list of all SparseML modifiers in the managed recipe that manage
            model sparsity
        """
        return [
            mod
            for mod in self.iter_modifiers()
            if SparsificationTypes.pruning in mod.sparsification_types
        ]

    @ModifierProp(serializable=False)
    def quantization_modifiers(self) -> List[BaseModifier]:
        """
        :return: list of all SparseML modifiers in the managed recipe that manage
            model quantization
        """
        return [
            mod
            for mod in self.iter_modifiers()
            if SparsificationTypes.quantization in mod.sparsification_types
        ]

    @ModifierProp(serializable=False)
    def distillation_modifiers(self) -> List[BaseModifier]:
        """
        :return: list of all SparseML modifiers in the managed recipe that manage
            Distillation
        """
        return [
            mod
            for mod in self.iter_modifiers()
            if SparsificationTypes.distillation in mod.sparsification_types
        ]

    @ModifierProp(serializable=False)
    def structured_modifiers(self) -> List[BaseModifier]:
        """
        :return: list of all SparseML modifiers in the managed recipe that manage
            structure changes to a model such as layer pruning, fitler pruning,
            and quantization
        """
        return [
            mod
            for mod in self.iter_modifiers()
            if SparsificationTypes.structured in mod.sparsification_types
        ]

    @ModifierProp(serializable=False)
    def min_epochs(self) -> int:
        """
        :return: the minimum epochs required by any of the modifiers under the manager
        """
        vals = []
        vals.extend(
            [
                math.floor(mod.start_epoch)
                for mod in self.iter_modifiers()
                if mod.start_epoch > -1
            ]
        )
        vals.extend(
            [
                math.floor(mod.end_epoch)
                for mod in self.iter_modifiers()
                if mod.end_epoch > -1
            ]
        )

        return min(vals) if len(vals) > 0 else -1

    @ModifierProp(serializable=False)
    def max_epochs(self) -> int:
        """
        :return: the maximum number of epochs required by any of the modifiers
            under the manager
        """
        vals = []
        vals.extend(
            [
                math.ceil(mod.start_epoch)
                for mod in self.iter_modifiers()
                if mod.start_epoch > -1
            ]
        )
        vals.extend(
            [
                math.ceil(mod.end_epoch)
                for mod in self.iter_modifiers()
                if mod.end_epoch > -1
            ]
        )

        return max(vals) if len(vals) > 0 else -1

    def save(self, file_path: str):
        """
        :param file_path: the file path to save the yaml config representation to
        """
        file_path = clean_path(file_path)
        create_parent_dirs(file_path)

        with open(file_path, "w") as yaml_file:
            yaml_file.write(str(self))

    def finalize_and_save_structured_modifiers(self, file_path: str):
        """
        saves a recipe containing only the structure modifiers of this
        manager. start and end epochs are overwritten so that they will
        be applied by epoch 0 in order

        :param file_path: file path to save the yaml recipe to
        """
        structured_modifiers = [deepcopy(mod) for mod in self.structured_modifiers]
        min_epoch = (-1.0 * len(structured_modifiers)) - 1
        for mod in structured_modifiers:
            if hasattr(mod, "start_epoch"):
                mod.start_epoch = min_epoch
            if hasattr(mod, "end_epoch"):
                mod.end_epoch = min_epoch
            min_epoch += 1

        structured_stage = {"structured_initialize_stage": structured_modifiers}
        structured_recipe_lines = self.modifiers_list_to_string_lines(structured_stage)
        structured_recipe_yaml = "\n".join(structured_recipe_lines)

        file_path = clean_path(file_path)
        create_parent_dirs(file_path)

        with open(file_path, "w") as yaml_file:
            yaml_file.write(structured_recipe_yaml)

    def iter_modifiers(self) -> Generator[None, None, BaseModifier]:
        """
        :return: generator for modifiers of this manager
        """
        modifiers_dict = (
            {"": self._modifiers}
            if isinstance(self._modifiers, List)
            else self._modifiers
        )
        for modifiers_list in modifiers_dict.values():
            for mod in modifiers_list:
                yield mod

    def to_string_lines(self) -> List[str]:
        """
        :return: a list of lines for a string / yaml representation of this instance
        """
        yaml_str_lines = ["version: 1.1.0", ""]
        if isinstance(self.modifiers, List):
            yaml_str_lines.append("modifiers:")
        yaml_str_lines.extend(self.modifiers_to_string_lines(self.modifiers))

        return yaml_str_lines

    def modifiers_to_string_lines(
        self, modifiers: Union[List[BaseModifier], Dict[str, List[BaseModifier]]]
    ) -> List[str]:
        """
        :param modifiers: the modifiers to convert into string / yaml representation
            for within the manage
        :return: a list of lines for a string / yaml representation of the
            modifiers in the manager
        """
        if isinstance(modifiers, List):
            return self.modifiers_list_to_string_lines(modifiers)

        yaml_str_lines = []
        for stage, stage_modifiers in modifiers.items():
            # stage name for yaml dict
            yaml_str_lines.append(f"{stage}:")
            # put all modifiers in stage into single modifier group
            yaml_str_lines.append(f"  {stage}_modifiers:")
            stage_yaml_str_lines = self.modifiers_list_to_string_lines(stage_modifiers)
            for stage_yaml_line in stage_yaml_str_lines:
                # add indentation to each modifier yaml str
                yaml_str_lines.append(f"  {stage_yaml_line}")
            # add blank line
            yaml_str_lines.append("")
        return yaml_str_lines

    def modifiers_list_to_string_lines(
        self, modifiers: List[BaseModifier]
    ) -> List[str]:
        """
        :param modifiers: the modifiers to convert into string / yaml representation
            for within the manage
        :return: a list of lines for a string / yaml representation of the
            modifiers in the manager
        """
        yaml_str_lines = []

        for mod in modifiers:
            mod_yaml = str(mod)
            mod_yaml_lines = mod_yaml.splitlines()

            for index, line in enumerate(mod_yaml_lines):
                if index == 0:
                    yaml_str_lines.append("    - {}".format(line))
                else:
                    yaml_str_lines.append("    {}".format(line))

            yaml_str_lines.append("")

        return yaml_str_lines

    def qat_active(self, epoch: float) -> bool:
        """
        :param epoch: the epoch to check if quantization aware training will be
            active during
        :return: True if quantization aware training will be active at the start
            of or within the given epoch, False otherwise
        """
        quant_modifiers = self.quantization_modifiers

        return (
            min(mod.start_epoch for mod in quant_modifiers) < epoch + 1
            if quant_modifiers
            else False
        )


def _sort_modifiers_list(modifiers: List[BaseModifier]) -> List[BaseModifier]:
    return sorted(modifiers, key=cmp_to_key(BaseModifier.comparator))
