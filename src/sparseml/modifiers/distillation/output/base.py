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

from typing import Any, Dict, List, Tuple, Union

from sparseml.core import Modifier, State


__all__ = ["OutputDistillationModifier"]


class OutputDistillationModifier(Modifier):
    targets: Union[str, List[Union[str, Tuple[str, str]]]]
    projection: str = None
    projection_args: Dict[str, Any] = None
    transforms: Union[str, List[str]] = "softmax"
    transforms_args: Union[Dict[str, Any], List[Dict[str, Any]]] = None
    comparison: str = "kl_divergence"
    comparison_args: Dict[str, Any] = None
    orig_scale: float = 1.0
    distill_scale: float = 1.0

    def on_initialize_structure(self, state: State, **kwargs):
        pass  # nothing needed for this modifier
