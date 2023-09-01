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


from enum import Enum
import importlib

__all__ = ["Framework", "MultiFrameworkObject"]


class Framework(Enum):
    general = "general"
    pytorch = "pytorch"
    tensorflow = "tensorflow"
    onnx = "onnx"
    keras = "keras"
    jax = "jax"

    @classmethod
    def from_str(cls, framework: str) -> "Framework":
        framework = framework.lower().strip()
        if framework == "general":
            return cls.general
        if framework == "pytorch":
            return cls.pytorch
        if framework == "tensorflow":
            return cls.tensorflow
        if framework == "onnx":
            return cls.onnx
        if framework == "keras":
            return cls.keras
        if framework == "jax":
            return cls.jax
        raise ValueError(f"Unknown framework: {framework}")

    def __str__(self):
        return self.value

    def formatted(self) -> str:
        if self == self.general:
            return "General"
        if self == self.pytorch:
            return "PyTorch"
        if self == self.tensorflow:
            return "TensorFlow"
        if self == self.onnx:
            return "ONNX"
        if self == self.keras:
            return "Keras"
        if self == self.jax:
            return "JAX"
        raise ValueError(f"Unknown framework: {self}")

    def class_name(self) -> str:
        return self.formatted() if self != self.general else ""


class MultiFrameworkObject:
    def __new__(
        cls,
        framework: Framework = None,
        enable_experimental: bool = False,
        **kwargs,
    ):
        if cls is MultiFrameworkObject:
            raise TypeError("MultiFrameworkObject cannot be instantiated directly")

        instance = super(MultiFrameworkObject, cls).__new__(cls, **kwargs)

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
        module = importlib.import_module(package)

        return getattr(module, class_name)
