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
Modifier for models through quantization aware training.

PyTorch version must support quantization (>=1.2, ONNX export support introduced in 1.7)
"""


from typing import Any, Dict, Type

from sparseml.pytorch.sparsification.modifier import (
    PyTorchModifierYAML,
    ScheduledModifier,
)
from sparseml.pytorch.sparsification.quantization.legacy_modifier_quantization import (
    QuantizationModifier as LegacyQuantizationModifier,
)


__all__ = [
    "QuantizationModifier",
]


# do not move, required to be defined before PyTorchModifierYAML decorator
def _select_quantization_modifier(state: Dict[str, Any]) -> Type:
    # if kwargs for the legacy quantization modifier are provided,
    # route YAML loading to that class
    return LegacyQuantizationModifier if "submodules" in state else QuantizationModifier


@PyTorchModifierYAML(swap_class_by_state_fn=_select_quantization_modifier)
class QuantizationModifier(ScheduledModifier):
    pass
