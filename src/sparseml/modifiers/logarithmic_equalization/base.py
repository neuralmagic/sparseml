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

from sparseml.modifiers.smoothquant import SmoothQuantModifier

__all__ = ["LogarithmicEqualizationModifier"]


class LogarithmicEqualizationModifier(SmoothQuantModifier):
    """
     Implements the Logarithmic Equalization Algorithm from https://arxiv.org/abs/2308.15987.
     This modifier performs a channel-wise smoothing of outliers in activations, making them
     easier to quantize by reducing the dynamic range. The smoothing is offset by
     applying the inverse operation to the next layer of weights, making the weights
     slightly more difficult to quantize.

     Because this modifier manipulates the weights of the model, it can only be used
     in one-shot and not during training. Activation ranges are determined by running a
     small set of calibration data through the model.

     This algorithm is very similar to SmoothQuant, changing only how the smoothing scales
     are computed. This modifier inherits most functionality from the SmoothQuantModifier.

    example recipe:
     ```yaml
     LogarithmicEqualizationModifier:
       mappings: [
         [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*self_attn_layer_norm"],
         [["re:.*fc1"], "re:.*final_layer_norm"]
       ]
       ignore: ["model.decoder.final_layer_norm"]
     ```

     :param mappings: list activation layers to smooth, and the which layers to offset
     the smoothing to for each activation
     :param ignore: list of layers to ignore, even if they match a regex in mappings
     :param num_calibration_steps: number of samples to use for calibration, or None to
     use the whole dataset
    """