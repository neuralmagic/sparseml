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
from typing import Optional

from sparseml.core.framework import Framework


__all__ = ["MultiFrameworkObject"]


class MultiFrameworkObject:
    """
    A base class for objects that can be instantiated for different ML frameworks.

    Subclassing this will allow loading framework specific implementations
    of the subclassed object, if they exist. If they do not exist, the subclass
    instance will be returned directly
    """

    def __new__(
        cls,
        framework: Optional[Framework] = None,
        enable_experimental: bool = False,
        **kwargs,
    ):
        if cls is MultiFrameworkObject:
            raise TypeError("MultiFrameworkObject cannot be instantiated directly")

        instance = super(MultiFrameworkObject, cls).__new__(cls)

        package = instance.__class__.__module__.rsplit(".", 1)[0]
        class_name = instance.__class__.__name__

        if framework is None or framework == Framework.general:
            return instance

        if enable_experimental:
            # check under the experimental package first
            try:
                return MultiFrameworkObject.load_framework_class(
                    f"{package}.experimental.{str(framework)}",
                    f"{class_name}{framework.class_name()}",
                )(**kwargs)
            except ImportError:
                pass

        # next check under the main package for the framework version
        try:
            return MultiFrameworkObject.load_framework_class(
                f"{package}.{str(framework)}", f"{class_name}{framework.class_name()}"
            )(**kwargs)
        except ImportError:
            pass

        # fall back on the class that was requested and
        # fail later if it doesn't support that framework
        return instance

    @staticmethod
    def load_framework_class(package: str, class_name: str):
        """
        Load the class from the given package and class name.

        :param package: The package to load the class from
        :param class_name: The name of the class to load
        :return: The loaded class module
        """
        module = importlib.import_module(package)

        return getattr(module, class_name)
