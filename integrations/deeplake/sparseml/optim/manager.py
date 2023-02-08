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

import json
import logging
import math
from collections import OrderedDict
from copy import deepcopy
from functools import cmp_to_key
from typing import Any, Dict, Generator, List, Optional, Union

from sparseml.optim.modifier import BaseModifier, BaseObject, ModifierProp
from sparseml.sparsification.types import SparsificationTypes
from sparseml.utils import RECIPE_METADATA_KEY, clean_path, create_parent_dirs


__all__ = ["BaseManager"]

_LOGGER = logging.getLogger(__name__)


class BaseManager(BaseObject):
    """
    Parent class meant to be used for all managers.
    Handles base implementations for properties and methods.

    :param modifiers: the modifiers to wrap
    :metadata: additional (to the information provided in the recipe) data to be
        preserved and possibly utilized - for reproducibility and completeness
    """

    def __init__(
        self,
        modifiers: Union[List[BaseModifier], Dict[str, List[BaseModifier]]],
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._metadata = metadata if metadata else None
        if self._metadata is not None:
            self._info_log_metadata()

        if isinstance(modifiers, List):
            # sort modifiers by when they start and end so that later modifiers
            # can overwrite in a deterministic order such as when initializing
            self._modifiers = self._sort_modifiers_list(modifiers)
        elif isinstance(modifiers, Dict):
            # staged recipe
            # sort modifiers of each stage by start/end as above then sort stages
            # by their modifiers
            modifiers = {
                stage: self._sort_modifiers_list(stage_modifiers)
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

    @staticmethod
    def _sort_modifiers_list(modifiers: List[BaseModifier]) -> List[BaseModifier]:
        return sorted(modifiers, key=cmp_to_key(BaseModifier.comparator))

    @property
    def metadata(self):
        return self._metadata

    def num_stages(self) -> int:
        """
        Return the number of stages of the recipe
        :return: number of stages
        """
        if isinstance(self.modifiers, dict):
            return len(self.modifiers)
        else:
            return 1

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    @classmethod
    def compose_staged(
        cls,
        base_recipe: Union[str, "BaseManager"],
        additional_recipe: Union[str, "BaseManager"],
        keep_original_epochs: bool = False,
        save_path: Optional[str] = None,
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
        :param save_path: optional path string; if provided, will be used to
            immediately save the combined multi-stage recipe to yaml
        :return: framework Manager object with the loaded composed recipe
        """

        # will load using class implementation of from_yaml
        # will fail from BaseModifier
        if isinstance(base_recipe, BaseManager):
            base_recipe = str(base_recipe)
        base_recipe = cls.from_yaml(base_recipe)

        if isinstance(additional_recipe, BaseManager):
            additional_recipe = str(additional_recipe)
        additional_recipe = cls.from_yaml(additional_recipe)

        # Both base_recipe and additional_recipe are non-staged_recipes
        if isinstance(base_recipe.modifiers, List) and isinstance(
            additional_recipe.modifiers, List
        ):
            # Need to generate stage names for two standard recipes
            base_stage_name, additional_stage_name = "stage_0", "stage_1"

            base_stages = {base_stage_name: base_recipe.modifiers}
            additional_stages = {additional_stage_name: additional_recipe.modifiers}

            base_recipe.metadata[base_stage_name] = base_recipe.metadata.pop(
                RECIPE_METADATA_KEY
            )
            additional_recipe.metadata[
                additional_stage_name
            ] = additional_recipe.metadata.pop(RECIPE_METADATA_KEY)

        # Base_recipe is staged recipe and additional_recipe is not
        elif isinstance(base_recipe.modifiers, OrderedDict) and isinstance(
            additional_recipe.modifiers, List
        ):

            base_stages = base_recipe.modifiers

            additional_stage_name = f"stage_{len(base_stages) + 1}"
            if additional_stage_name in base_stages.keys():
                raise ValueError(
                    f"Generated new stage name: {additional_stage_name}, "
                    "but there already exists"
                    "a stage with that name in the checkpoint file. "
                    "Please edit the stage name in the checkpoint file."
                )

            additional_stages = {additional_stage_name: additional_recipe.modifiers}

            additional_recipe.metadata[
                additional_stage_name
            ] = additional_recipe.metadata.pop(RECIPE_METADATA_KEY)

        # Additional_recipe is staged recipe and base_recipe is not
        elif isinstance(base_recipe.modifiers, List) and isinstance(
            additional_recipe.modifiers, OrderedDict
        ):
            additional_stages = additional_recipe.modifiers

            base_stage_name = f"pre_{list(additional_stages.keys())[0]}"

            base_stages = {base_stage_name: base_recipe.modifiers}

            base_recipe.metadata[base_stage_name] = base_recipe.metadata.pop(
                RECIPE_METADATA_KEY
            )

        # Both recipes are staged.
        else:
            base_stages = base_recipe.modifiers
            additional_stages = additional_recipe.modifiers

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

            # make sure that for the modifiers in base_stages
            # with the initial attribute `end_epoch` = -1,
            # this attribute value is replaced with `base_end_epoch`
            for base_modifiers in base_stages.values():
                for base_modifier in base_modifiers:
                    if (
                        hasattr(base_modifier, "end_epoch")
                        and base_modifier.end_epoch == -1
                    ):
                        base_modifier._init_end = base_end_epoch
                        base_modifier.end_epoch = base_end_epoch

            for additional_modifiers in additional_stages.values():
                for additional_modifier in additional_modifiers:
                    additional_modifier.advance_epochs(ref_start_epoch=base_end_epoch)

        combined_stages = base_stages
        combined_stages.update(additional_stages)

        combined_metadata = base_recipe.metadata
        combined_metadata.update(additional_recipe.metadata)

        combined_manager = cls(combined_stages, combined_metadata)

        if save_path:
            combined_manager.save(save_path)

        return combined_manager

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

    def save(self, file_path: str, include_metadata: bool = True):
        """
        :param file_path: the file path to save the yaml config representation to
        :param include_metadata: boolean indicator whether metadata shall be
            appended to the yaml file before saving. Default is True.
        """
        file_path = clean_path(file_path)
        create_parent_dirs(file_path)

        with open(file_path, "w") as yaml_file:
            yaml_file.write("\n".join(self.to_string_lines(include_metadata)))

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

    def to_string_lines(self, include_metadata: bool = True) -> List[str]:
        """
        :param include_metadata: boolean indicator whether metadata shall be
            appended to the yaml file before saving. Default is False.
        :return: a list of lines for a string / yaml representation of this instance
        """
        yaml_str_lines = ["version: 1.1.0", ""]
        # parse standard recipe
        if isinstance(self.modifiers, List):
            if include_metadata and self._metadata:
                yaml_str_lines.extend(self.metadata_to_string_lines())
            yaml_str_lines.append("modifiers:")
            yaml_str_lines.extend(self.modifiers_list_to_string_lines(self.modifiers))
        # parse staged recipe
        else:
            yaml_str_lines.extend(
                self.modifiers_to_string_lines(self.modifiers, include_metadata)
            )

        return yaml_str_lines

    def metadata_to_string_lines(self, stage: str = None) -> List[str]:
        """
        Parse `self._metadata` into list of strings.
        :param stage: Name of the current recipe stage.
            If stage = None, we are dealing with standard, unstaged recipe.
        :return: a list of lines for a string / yaml representation of the
            metadata for the given stage in the manager
        """
        yaml_str_lines = []

        if stage:
            yaml_str_lines.append(f"  {RECIPE_METADATA_KEY}:")
            if not isinstance(self._metadata[stage], dict):
                yaml_str_lines[-1] += f" {self._metadata[stage]}"
            else:
                yaml_str_lines = _nested_dict_to_lines(
                    self._metadata[stage], yaml_str_lines, nesting_depth=2
                )

        else:
            yaml_str_lines.append(f"{RECIPE_METADATA_KEY}:")
            if not isinstance(self._metadata, dict):
                yaml_str_lines[-1] += f" {self._metadata}"
            else:
                yaml_str_lines = _nested_dict_to_lines(
                    self._metadata[RECIPE_METADATA_KEY], yaml_str_lines
                )

        yaml_str_lines.append("")
        return yaml_str_lines

    def modifiers_to_string_lines(
        self,
        modifiers: Union[List[BaseModifier], Dict[str, List[BaseModifier]]],
        include_metadata: bool = True,
    ) -> List[str]:
        """
        :param modifiers: the modifiers to convert into string / yaml representation
            for within the manage
        :param include_metadata: boolean indicator whether metadata shall be
            appended to the yaml file before saving.
        :return: a list of lines for a string / yaml representation of the
            modifiers in the manager
        """

        yaml_str_lines = []
        for stage, stage_modifiers in modifiers.items():
            # stage name for yaml dict
            yaml_str_lines.append(f"{stage}:")

            if include_metadata and self._metadata:
                yaml_str_lines.extend(self.metadata_to_string_lines(stage))

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

    def _info_log_metadata(self):
        metadata_str = json.dumps(self._metadata, indent=1)
        _LOGGER.debug(f"Created recipe manager with metadata: {metadata_str}")


def _nested_dict_to_lines(
    dict1: dict, yaml_str_lines: List[str], nesting_depth: int = 1
) -> List[str]:
    indentation = "  "

    if dict1 is None:
        return yaml_str_lines

    for key, value in dict1.items():
        if isinstance(value, dict):
            # add data for the current nesting level and
            # move deeper to the next nesting level
            yaml_str_lines.append(indentation * nesting_depth + f"{key}:")
            yaml_str_lines = _nested_dict_to_lines(
                value, yaml_str_lines, nesting_depth + 1
            )
        else:
            # reached maximum nesting level.
            yaml_str_lines.append(indentation * nesting_depth + f"{key}: {value}")
    return yaml_str_lines
