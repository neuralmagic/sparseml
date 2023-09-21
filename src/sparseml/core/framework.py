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
from enum import Enum

from pydantic import ValidationError


__all__ = ["Framework"]


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
