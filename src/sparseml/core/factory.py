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


import importlib
import pkgutil
from typing import Dict, Type

from sparseml.core.framework import Framework
from sparseml.core.modifier import Modifier


__all__ = ["ModifierFactory"]


class ModifierFactory:
    """
    A factory for loading and registering modifiers

    :param _MAIN_PACKAGE_PATH: The path to the main modifiers package
    :param _EXPERIMENTAL_PACKAGE_PATH: The path to the experimental modifiers package
    :param _loaded: Whether or not the factory has been loaded
    :param _main_registry: The registry of main modifiers
    :param _experimental_registry: The registry of experimental modifiers
    :param _registered_registry: The registry of registered modifiers
    :param _errors: The errors that occurred when loading the modifiers
    """

    _MAIN_PACKAGE_PATH = "sparseml.modifiers"
    _EXPERIMENTAL_PACKAGE_PATH = "sparseml.modifiers.experimental"

    _loaded: bool = False
    _main_registry: Dict[str, Type[Modifier]] = {}
    _experimental_registry: Dict[str, Type[Modifier]] = {}
    _registered_registry: Dict[str, Type[Modifier]] = {}
    _errors: Dict[str, Exception] = {}

    @staticmethod
    def refresh():
        """
        A method to refresh the factory by reloading the modifiers
        Note: this will clear any previously registered modifiers
        """
        ModifierFactory._main_registry = ModifierFactory.load_from_package(
            ModifierFactory._MAIN_PACKAGE_PATH
        )
        ModifierFactory._experimental_registry = ModifierFactory.load_from_package(
            ModifierFactory._EXPERIMENTAL_PACKAGE_PATH
        )
        ModifierFactory._loaded = True

    @staticmethod
    def load_from_package(package_path: str) -> Dict[str, Type[Modifier]]:
        """
        :param package_path: The path to the package to load modifiers from
        :return: The loaded modifiers, as a mapping of name to class
        """
        loaded = {}
        main_package = importlib.import_module(package_path)

        for importer, modname, is_pkg in pkgutil.walk_packages(
            main_package.__path__, package_path + "."
        ):
            try:
                module = importlib.import_module(modname)

                for attribute_name in dir(module):
                    if not attribute_name.endswith("Modifier"):
                        continue

                    try:
                        if attribute_name in loaded:
                            continue

                        attr = getattr(module, attribute_name)

                        if not isinstance(attr, type):
                            raise ValueError(
                                f"Attribute {attribute_name} is not a type"
                            )

                        if not issubclass(attr, Modifier):
                            raise ValueError(
                                f"Attribute {attribute_name} is not a Modifier"
                            )

                        loaded[attribute_name] = attr
                    except Exception as err:
                        # TODO: log import error
                        ModifierFactory._errors[attribute_name] = err
            except Exception as module_err:
                # TODO: log import error
                print(module_err)

        return loaded

    @staticmethod
    def create(
        type_: str,
        framework: Framework,
        allow_registered: bool,
        allow_experimental: bool,
        **kwargs,
    ) -> Modifier:
        """
        Instantiate a modifier of the given type from registered modifiers.

        :raises ValueError: If no modifier of the given type is found
        :param type_: The type of modifier to create
        :param framework: The framework the modifier is for
        :param allow_registered: Whether or not to allow registered modifiers
        :param allow_experimental: Whether or not to allow experimental modifiers
        :param kwargs: Additional keyword arguments to pass to the modifier
            during instantiation
        :return: The instantiated modifier
        """
        if type_ in ModifierFactory._errors:
            raise ModifierFactory._errors[type_]

        if type_ in ModifierFactory._registered_registry:
            if allow_registered:
                return ModifierFactory._registered_registry[type_](
                    framework=framework, **kwargs
                )
            else:
                # TODO: log warning that modifier was skipped
                pass

        if type_ in ModifierFactory._experimental_registry:
            if allow_experimental:
                return ModifierFactory._experimental_registry[type_](
                    framework=framework, **kwargs
                )
            else:
                # TODO: log warning that modifier was skipped
                pass

        if type_ in ModifierFactory._main_registry:
            return ModifierFactory._main_registry[type_](framework=framework, **kwargs)

        raise ValueError(f"No modifier of type '{type_}' found.")

    @staticmethod
    def register(type_: str, modifier_class: Type[Modifier]):
        """
        Register a modifier class to be used by the factory.

        :raises ValueError: If the provided class does not subclass the Modifier
            base class or is not a type
        :param type_: The type of modifier to register
        :param modifier_class: The class of the modifier to register, must subclass
            the Modifier base class
        """
        if not issubclass(modifier_class, Modifier):
            raise ValueError(
                "The provided class does not subclass the Modifier base class."
            )
        if not isinstance(modifier_class, type):
            raise ValueError("The provided class is not a type.")

        ModifierFactory._registered_registry[type_] = modifier_class
