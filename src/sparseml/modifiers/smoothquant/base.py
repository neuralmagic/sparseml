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


from dataclasses import dataclass
from typing import Dict, Generic, List, Optional, Tuple, TypeVar

from sparseml.core import Modifier
from sparseml.core.model import ModifiableModel
from sparseml.core.model.base import LT
from sparseml.core.state import Event, State


VT = TypeVar("VT")  # represents a generic vector

__all__ = ["SmoothQuantScale", "SmoothQuantMapping", "SmoothQuantModifier"]


@dataclass
class SmoothQuantScale(Generic[VT]):
    """
    Dataclass for storing the channel-wise minimum and maximum values for a layer. This
    is updated each forward pass during calibration

    :param min_channel_vals: minimum output value seen so far, per channel
    :param max_channel_vals: maximum output value seen so far, per channel
    """

    min_channel_vals: VT
    max_channel_vals: VT


@dataclass
class SmoothQuantMapping(Generic[LT]):
    """
    Dataclass for storing the mapping between an activation layer and the following
    weights that must be balanced during smoothing

    :param smooth_name: name of the activation layer
    :param smooth_layer: PyTorch module storing the activation layer
    :param balance_layers: list of PyTorch modules that smooth_layer feeds into, must be
    balanced to offset the smoothing of smooth_layer
    """

    smooth_name: str
    smooth_layer: LT
    balance_layers: List[LT]


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

    example recipe:
     ```yaml
     SmoothQuantModifier:
       smoothing_strength: 0.5
       mappings: [
         [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*self_attn_layer_norm"],
         [["re:.*fc1"], "re:.*final_layer_norm"]
       ]
       ignore: ["model.decoder.final_layer_norm"]
     ```

     :param smoothing_strength: alpha, intensity of smoothing to perform (0-1 range)
     :param mappings: list activation layers to smooth, and which layers to
        scale the output such that activations are smoothed.
        Each entry of the mapping list should be a list itself, in which the first
        entry is a list of layers who share the same input activation (the one to be
        to smoothed) and the second entry is the layer whose output is scaled to
        achieve the smoothing.
        If regex is used, it matches layers with the largest overlap in module name.
     :param ignore: list of layers to ignore, even if they match a regex in mappings.
        It should match the name of layers whose outputs are scaled to achieve
        smoothing (the second entry of the mappings list).
     :param num_calibration_steps: number of samples to use for calibration, or None to
     use the whole dataset
    """

    smoothing_strength: float = 0.5
    mappings: List[Tuple]
    ignore: Optional[List[str]] = None
    num_calibration_steps: Optional[int] = None

    resolved_mappings_: Optional[List] = None
    scales_: Optional[Dict] = None

    def on_initialize_structure(self, state: State, **kwargs):
        pass  # nothing needed for this modifier

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Initialize and run SmoothQuant on the given state

        :param state: state to run SmoothQuant on
        :return: True on a successful run, False otherwise
        """
        if self.end and self.end != -1:
            raise ValueError(
                f"{self.__class__.__name__} can only be applied during one-shot. "
                f" Expected end to be None or -1, got {self.end}"
            )
        if self.start and self.start != -1:
            raise ValueError(
                f"{self.__class__.__name__} can only be applied during one-shot. "
                f"Expected start to be None or -1, got {self.end}"
            )

        self.ignore = [] if not self.ignore else self.ignore
        self.resolved_mappings_ = self._resolve_mappings(state.model)
        self.scales_ = {}

    def _resolve_mappings(self, model: ModifiableModel) -> List:
        """
        Transforms the list of activations to smooth and their corresponding weights
        into SmoothQuantMapping objects, resolving regular expressions.

        For each activation in the mapping list, we find the corresponding weight to
        balance by searching for the longest substring. For instance, if our balance
        weight is ".*re:.*q_proj" and the activation is "re:.*self_attn_layer_norm" we
        would match model.layer.0.p_proj to model.layer.0.self_attn_layer_norm and
        repeat for model.layer.1 and so on
        """
        resolved_mappings = []
        for to_balance, to_smooth in self.mappings:
            to_smooth_layers = model.get_layers(to_smooth)
            for layer_name, smooth_layer in to_smooth_layers.items():
                if layer_name not in self.ignore:
                    balance_layers = []
                    for balance_suffix in to_balance:
                        # find the submodule that matches the activation layer
                        _, balance_layer = model.get_matching_layer(
                            balance_suffix, layer_name, model.model
                        )
                        if balance_layer:
                            balance_layers.append(balance_layer)
                    # each mapping can contain multiple layers to balance, but only
                    # one layer to smooth
                    mapping = SmoothQuantMapping(
                        layer_name, smooth_layer, balance_layers
                    )
                    resolved_mappings.append(mapping)
        return resolved_mappings

    def on_start(self, state: State, event: Event, **kwargs):
        pass

    def on_update(self, state: State, event: Event, **kwargs):
        pass

    def on_end(self, state: State, event: Event, **kwargs):
        pass

    def on_event(self, state: State, event: Event, **kwargs):
        pass

    def on_finalize(self, state: State, **kwargs) -> bool:
        """
        Clean up by clearing the scale and mapping data

        :param state: unused
        :return: True
        """
        if self.scales_ is not None:
            self.scales_.clear()
        if self.resolved_mappings_ is not None:
            self.resolved_mappings_.clear()

        return True
