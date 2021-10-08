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
Classes for building YAML SparseML recipes without instantiating specific modifier\
implementations
"""


import textwrap
from typing import Any, Dict, List, Optional, Type, Union

import yaml

from sparseml.optim import BaseModifier, ModifierProp
from sparseml.utils import create_parent_dirs


__all__ = [
    "ModifierYAMLBuilder",
    "RecipeYAMLBuilder",
    "to_yaml_str",
]


class ModifierYAMLBuilder(object):
    """
    Class for building a YAML string representation of a modifier by setting
    various properties of it. Properties are automatically inferred through the
    serializable ModifierProps of the given modifier. They can be accessed through
    auto-generated set_{name} and get_{name}.

    :param modifier_class: reference to the class of modifier this object should create
        a YAML representation for
    """

    def __init__(self, modifier_class: Type[BaseModifier]):
        assert issubclass(
            modifier_class, BaseModifier
        ), "a subclass of Modifier must be used to instantiate a ModifierYAMLBuilder"

        self._modifier_class = modifier_class
        self._properties = {}

        for attr in dir(modifier_class):
            attr_obj = getattr(modifier_class, attr)
            if isinstance(attr_obj, ModifierProp) and attr_obj.serializable:
                self._add_setter_getter(attr)

    @property
    def modifier_class(self) -> Type[BaseModifier]:
        """
        :return: the class of the Modifier for which this object is building a string
        """
        return self._modifier_class

    def build_yaml_str(self) -> str:
        """
        :return: string representation of the built Modifier as a YAML list item
        """
        class_name_yaml = f"- !{self._modifier_class.__name__}"
        properties_yaml = "\n".join(
            [f"{key}: {to_yaml_str(value)}" for key, value in self._properties.items()]
        )
        properties_yaml = textwrap.indent(properties_yaml, "  ")
        return f"{class_name_yaml}\n{properties_yaml}"

    def _add_setter_getter(self, property_name: str):
        # define generic setter and getter functions for this property
        def setter(self_, val: Any) -> ModifierYAMLBuilder:
            self_._properties[property_name] = val
            return self_

        def getter(self_) -> Optional[Any]:
            return self_._properties.get(property_name)

        # bind setter and getter functions to this class
        setter_bound = setter.__get__(self, self.__class__)
        setattr(self, f"set_{property_name}", setter_bound)
        getter_bound = getter.__get__(self, self.__class__)
        setattr(self, f"get_{property_name}", getter_bound)


class RecipeYAMLBuilder(object):
    """
    Class for building a YAML SparseML recipe with standardized structure

    :param variables: dict of string initial variable names to non-modifier recipe
        variables to be included. Default is an empty dict
     :param modifier_groups: dict of string initial modifier group names to a list
        of ModifierYAMLBuilder objects of modifiers to be included in that group.
        All modifier group names must contain 'modifiers' in the string. Default is
        an empty dict
    """

    def __init__(
        self,
        variables: Dict[str, Any] = None,
        modifier_groups: Dict[str, List[ModifierYAMLBuilder]] = None,
    ):
        self._variables = variables or {}
        self._modifier_groups = modifier_groups or {}

        self._validate()

    def add_modifier_group(
        self, name: str, modifier_builders: List[ModifierYAMLBuilder] = None
    ) -> "RecipeYAMLBuilder":
        """
        Adds a modifier group with the given name to this builder

        :param name: name of new modifier group
        :param modifier_builders: list of modifier builder objects to initialize
            this group with. Default is an empty list
        :return: a reference to this object with the modifier group now added
        """
        self._validate_modifier_group_name(name)
        if name in self._modifier_groups:
            raise KeyError(
                f"{name} is already a modifier group name in this RecipeYAMLBuilder"
            )

        modifier_builders = modifier_builders or []
        self._modifier_groups[name] = modifier_builders
        self._validate()
        return self

    def get_modifier_group(self, name: str) -> Optional[List[ModifierYAMLBuilder]]:
        """
        :param name: name of the modifier group to retrieve the modifier builders of
        :return: reference to the list of modifier builders currently in this
            modifier group. if the modifier group does not exist, None will be
            returned
        """
        return self._modifier_groups.get(name)

    def get_modifier_builders(
        self,
        modifier_type: Optional[Union[Type[BaseModifier], str]] = None,
        modifier_groups: Optional[Union[List[str], str]] = None,
    ):
        """
        :param modifier_type: optional type of modifier to filter by. Can be
            a type reference that will match if the modifier is of that type
            or a subclass of it or a string where it will match if the class
            is exactly that name. Defaults to None
        :param modifier_groups: optional list of modifier group names to match
            to. Defaults to None
        :return: all modifier builders in this recipe, filtered by type and group
        """
        if isinstance(modifier_groups, str):
            modifier_groups = [modifier_groups]

        modifier_builders = []
        for group, builders in self._modifier_groups.items():
            if modifier_groups is not None and group not in modifier_groups:
                continue
            for builder in builders:
                if modifier_type and not self._modifier_builder_is_instance(
                    builder, modifier_type
                ):
                    continue
                modifier_builders.append(builder)
        return modifier_builders

    def get_variable(self, name: str, default: Any = None) -> Any:
        """
        :param name: name of the recipe variable to return
        :param default: default value that should be returned if the given
            name is not a current variable
        :return: current value of the given variable, or the default if
            the variable is not set in this builder
        """
        return self._variables.get(name, default)

    def has_variable(self, name: str) -> bool:
        """
        :param name: name of the recipe variable to check
        :return: True if this recipe builder has a variable with the given name.
            False otherwise
        """
        return name in self._variables

    def set_variable(self, name: str, val: Any) -> "RecipeYAMLBuilder":
        """
        Sets the given variable name to the given value

        :param name: variable name to set
        :param val: value to set the variable to
        :return: a reference to this object with the variable now set
        """
        self._variables[name] = val
        return self

    def build_yaml_str(self) -> str:
        """
        :return: yaml string representation of this recipe in standard format
        """
        # write variables
        yaml_str = "\n".join(
            [f"{key}: {to_yaml_str(value)}" for key, value in self._variables.items()]
        )
        # write modifier groups
        for group, builders in self._modifier_groups.items():
            if not builders:
                continue  # do not write empty groups
            modifiers_yaml = "\n\n".join(
                [builder.build_yaml_str() for builder in builders]
            )
            modifiers_yaml = textwrap.indent(modifiers_yaml, "  ")
            yaml_str += f"\n\n{group}:\n{modifiers_yaml}"
        return yaml_str

    def save_yaml(self, file_path: str):
        """
        Saves this recipe as a yaml file to the given path

        :param file_path: file path to save file to. if no '.' character is found
            in the path, '.yaml' will be added to the path
        """
        if "." not in file_path:
            file_path += ".yaml"
        self._save_file_str(self.build_yaml_str(), file_path)

    def save_markdown(self, file_path: str, desc: str = ""):
        """
        Saves this recipe as a markdown file to the given path with the
        recipe yaml contained in the frontmatter

        :param file_path: file path to save file to. if no '.' character is found
            in the path, '.md' will be added to the path
        :param desc: optional description to add to the markdown file after the recipe
            YAML in the frontmatter. Default is empty string
        """
        if "." not in file_path:
            file_path += ".md"

        md_content = f"---\n{self.build_yaml_str()}\n---\n{desc}"
        self._save_file_str(md_content, file_path)

    @staticmethod
    def _save_file_str(content: str, file_path: str):
        create_parent_dirs(file_path)
        with open(file_path, "w") as file:
            file.write(content)

    @staticmethod
    def _validate_modifier_group_name(name: str) -> bool:
        if "modifiers" not in name:
            raise ValueError(
                "modifier groups must contain 'modifiers' in their name received "
                f"group with name: {name}"
            )

    @staticmethod
    def _modifier_builder_is_instance(
        builder: ModifierYAMLBuilder, type_: Union[Type[BaseModifier], str]
    ) -> bool:
        builder_class = builder.modifier_class
        if isinstance(type_, str):
            return builder_class.__name__ == type_
        return builder_class is type_ or issubclass(builder_class, type_)

    def _validate(self):
        if not isinstance(self._variables, Dict):
            raise ValueError(
                "RecipeYAMLBuilder variables object must be a Dict "
                f"found type {type(self._variables)}"
            )

        for name, builders in self._modifier_groups.items():
            self._validate_modifier_group_name(name)

            if not isinstance(builders, List):
                raise ValueError(
                    "All modifier groups in RecipeYAMLBuilder must contain a list"
                    f"of ModifierYAMLBuilder objects. Group {name} has value of "
                    f"type {type(builders)}"
                )

            for builder in builders:
                if not isinstance(builder, ModifierYAMLBuilder):
                    raise ValueError(
                        "All modifier groups in RecipeYAMLBuilder must contain a "
                        f"list of ModifierYAMLBuilder objects. Group {name} "
                        f"contains an element of type {type(builder)}"
                    )


def to_yaml_str(val: Any) -> str:
    """
    :param val: value to get yaml str value of
    :return: direct str cast of val if it is an int, float, or bool, otherwise
        the stripped output of yaml.dump
    """
    if isinstance(val, (str, int, float, bool)):
        return str(val)
    else:
        return yaml.dump(val).strip()
