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

import torch.nn as nn

from ultralytics.nn import modules as ulm


class Conv(nn.Module):
    """
    Slightly modified version of ultralytics Conv with SiLU instantiated
    for each instance. This is to help with SiLU naming in SparseML recipe
    """

    def __init__(self, layer: ulm.Conv):
        super().__init__()
        self.conv = layer.conv
        self.bn = layer.bn
        for attr in ["i", "f", "type"]:
            if hasattr(layer, attr):
                setattr(self, attr, getattr(layer, attr))
        is_silu = isinstance(layer.act, nn.SiLU)
        self.act = nn.SiLU() if is_silu else layer.act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class AddInput(nn.Module):
    """
    Equivalent to Identity for quantization support
    """

    def forward(self, x):
        return x


class Bottleneck(nn.Module):
    """
    Modified version of ultralyltics Bottleneck with inputs of the residual
    adds being marked for potential quantization
    """

    def __init__(self, layer: ulm.Bottleneck):
        super().__init__()
        self.cv1 = layer.cv1
        self.cv2 = layer.cv2
        self.add = layer.add
        for attr in ["i", "f", "type"]:
            if hasattr(layer, attr):
                setattr(self, attr, getattr(layer, attr))
        self.add_input_0 = AddInput()
        self.add_input_1 = AddInput()

    def forward(self, x):
        return (
            self.add_input_0(x) + self.add_input_1(self.cv2(self.cv1(x)))
            if self.add
            else self.cv2(self.cv1(x))
        )
