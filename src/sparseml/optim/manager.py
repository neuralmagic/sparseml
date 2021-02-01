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
from typing import List

from sparseml.optim.modifier import BaseObject, BaseScheduled, ModifierProp
from sparseml.utils import clean_path, create_parent_dirs


__all__ = ["BaseManager"]


class BaseManager(BaseObject):
    """
    Parent class meant to be used for all managers.
    Handles base implementations for properties and methods.

    :param modifiers: the modifiers to wrap
    """

    def __init__(self, modifiers: List[BaseScheduled], **kwargs):
        super().__init__(**kwargs)
        # sort the modifiers so they are iterated in order of their start epoch
        # if start epoch is the same, end epoch is used to break ties
        # with ending first running first
        self._modifiers = sorted(
            modifiers, key=lambda m: m.start_epoch + m.end_epoch * 1e-6
        )

    def __del__(self):
        for mod in self._modifiers:
            del mod

        self._modifiers.clear()

    def __str__(self) -> str:
        return "\n".join(self.to_string_lines())

    @ModifierProp(serializable=False)
    def modifiers(self) -> List[BaseScheduled]:
        return self._modifiers

    @ModifierProp(serializable=False)
    def min_epochs(self) -> int:
        """
        :return: the minimum epochs required by any of the modifiers under the manager
        """
        vals = []
        vals.extend(
            [
                math.floor(mod.start_epoch)
                for mod in self._modifiers
                if mod.start_epoch > -1
            ]
        )
        vals.extend(
            [math.floor(mod.end_epoch) for mod in self._modifiers if mod.end_epoch > -1]
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
                for mod in self._modifiers
                if mod.start_epoch > -1
            ]
        )
        vals.extend(
            [math.ceil(mod.end_epoch) for mod in self._modifiers if mod.end_epoch > -1]
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

    def to_string_lines(self) -> List[str]:
        """
        :return: a list of lines for a string / yaml representation of this instance
        """
        yaml_str_lines = ["version: 1.1.0", "", "modifiers:"]
        yaml_str_lines.extend(self.modifiers_to_string_lines(self.modifiers))

        return yaml_str_lines

    def modifiers_to_string_lines(self, modifiers: List[BaseScheduled]) -> List[str]:
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
