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

import logging

from sparseml.core import Event, State
from sparseml.modifiers.smoothquant.base import SmoothQuantModifier


_LOGGER = logging.getLogger(__name__)


class SmoothQuantModifierPyTorch(SmoothQuantModifier):
    qat_enabled_: bool = False

    def on_initialize(self, state: State, **kwargs) -> bool:
        pass

    def on_finalize(self, state: State, **kwargs) -> bool:
        pass

    def on_start(self, state: State, event: Event, **kwargs):
        pass

    def on_update(self, state: State, event: Event, **kwargs):
        pass

    def on_end(self, state: State, event: Event, **kwargs):
        pass

    def on_event(self, state: State, event: Event, **kwargs):
        pass
