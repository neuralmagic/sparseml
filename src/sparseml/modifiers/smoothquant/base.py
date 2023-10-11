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


from typing import List, Optional, Tuple

from sparseml.core import Modifier
from sparseml.core.state import Event, State


__all__ = ["SmoothQuantModifier"]


class SmoothQuantModifier(Modifier):
    """
    Implements the SmoothQuant algorithm from https://arxiv.org/abs/2211.10438. This
    modifier performs a channel-wise smoothing of outliers in activations, making them
    easier to quantize by reducing the dynamic range. The smoothing is offset by
    applying the inverse operation to the next layer of weights, making the weights
    slightly more difficult to quantize.

    Because this modifier manipulates the weights of the model, it can only be used in
    in one-shot and not during training. Activation ranges are determined by running a
    small set of calibration data through the model.

    :param migration_strength: Intensity of smoothing to perform (0-1 range)
    :param mappings: list activation layers to smooth, and the which layers to offset
    the smoothing to for each activation
    :param ignore: list of layers to ignore, even if they match a regex in mappings
    :param logarithmic_equalization: Whether to use a logarithmic scale for smoothing
    :param num_calibration_steps: number of samples to use for calibration, or None to
    use the whole dataset
    """

    migration_strength: float
    mappings: List[Tuple]
    ignore: Optional[List[str]] = None
    logarithmic_equalization: Optional[bool] = False
    num_calibration_steps: Optional[int] = None

    def on_initialize_structure(self, state: "State", **kwargs):
        pass  # nothing needed for this modifier

    def on_start(self, state: State, event: Event, **kwargs):
        pass

    def on_update(self, state: State, event: Event, **kwargs):
        pass

    def on_end(self, state: State, event: Event, **kwargs):
        pass

    def on_event(self, state: State, event: Event, **kwargs):
        pass
