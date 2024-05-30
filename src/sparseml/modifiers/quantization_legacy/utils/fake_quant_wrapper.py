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

import torch
from packaging import version


_PARSED_TORCH_VERSION = version.parse(torch.__version__)
if _PARSED_TORCH_VERSION < version.parse("2"):
    from torch.quantization import FakeQuantize
else:
    from torch.ao.quantization import FakeQuantize


class FakeQuantizeWrapper(FakeQuantize):
    """
    Wrapper around Pytorch's FakeQuantize module, to enable compatibility with bfloat16
    and auto device mapping.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Overrides the forward pass by converting the weight to a compatible dtype and
        moving configuration parameters to the device of the weight.

        All changes are reverted before returning.
        """
        X = args[0]
        if self.fake_quant_enabled[0] == 0:
            return super().forward(X)

        # this assumes all params are on the same device
        og_params_device = self.zero_point.device
        og_weight_dtype = X.dtype

        # setup for FakeQuantize operation
        if og_params_device != X.device:
            self._move_params_to_device(X.device)
        if og_weight_dtype is torch.bfloat16:
            X = X.to(torch.float32)

        X = super().forward(X)

        # move params back to where they were
        if og_params_device != X.device:
            self._move_params_to_device(og_params_device)
        if og_weight_dtype is torch.bfloat16:
            X = X.to(og_weight_dtype)

        return X

    def _move_params_to_device(self, device: torch.device):
        """
        Moves parameters used in the FakeQuantize forward pass to device
        """
        self.zero_point = self.zero_point.to(device)
        self.scale = self.scale.to(device)

        if self.observer_enabled[0] == 0:
            return

        if hasattr(self.activation_post_process, "min_val"):
            self.activation_post_process.min_val = (
                self.activation_post_process.min_val.to(device)
            )
            self.activation_post_process.max_val = (
                self.activation_post_process.max_val.to(device)
            )
            self.activation_post_process.eps = self.activation_post_process.eps.to(
                device
            )
