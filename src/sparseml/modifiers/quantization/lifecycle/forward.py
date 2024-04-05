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

from functools import wraps

import torch
from torch.nn import Module

from sparseml.modifiers.quantization.lifecycle.status import QuantizationStatus


__all__ = ["wrap_module_forward_quantized"]


def quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    q_max: torch.Tensor,
) -> torch.Tensor:
    return torch.clamp(
        round(
            x / scale + zero_point,
            0,
            q_max,
        )
    )


def dequantize(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    return (x_q - zero_point) * scale


def fake_quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    args: QuantizationArgs,
) -> torch.Tensor:
    max_q = torch.tensor(2**args.n_bits - 1)
    columns = x.shape[1]
    Q = torch.zeros_like(x)
    for i1 in range(0, columns, args.block_size):
        i2 = min(i1 + args.block_size, columns)
        count = i2 - i1

        W1 = x[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)

        for i in range(count):
            w = W1[:, i]

            if args.group_size != -1:
                if (i1 + i) % args.group_size == 0:
                    xmin, xmax = get_qparams(
                        x[:, (i1 + i) : (i1 + i + args.group_size)], args.symmetric
                    )
                    scale, zero = get_scale_zero_point(
                        x[:, (i1 + i) : (i1 + i + args.group_size)],
                        max_q,
                        xmax,
                        xmin,
                        args.symmetric,
                        args.group_size,
                    )

            q = quantize(w.unsqueeze(1), scale, zero, max_q).flatten()
        Q1[:, i] = q
        Q[:, i1:i2] = Q1

    return dequantize(Q, scale, zero_point)


def wrap_module_forward_quantized(module: Module):
    # expects a module already initialized and injected with the parameters in
    # initialize_module_for_quantization
    forward_func_orig = module.forward.__func__

    @wraps(forward_func_orig)  # ensures docstring, names, etc are propagated
    def wrapped_forward(self, *args, **kwargs):
        input_ = args[0]

        if scheme.input_activations is not None:
            # calibrate and (fake) quantize input activations when applicable
            input_ = _maybe_calibrate_or_quantize(
                module, input_, "input", scheme.input_activations
            )

        if scheme.weights is not None:
            # calibrate and (fake) quantize weights when applicable
            self.weight.data = _maybe_calibrate_or_quantize(
                module, self.weight, "weight", scheme.weights
            )

        # perform wrapped forward call
        output = forward_func_orig.__get__(module, module.__class__)(
            input_, *args[1:], **kwargs
        )

        if scheme.output_activations is not None:
            # calibrate and (fake) quantize output activations when applicable
            output = _maybe_calibrate_or_quantize(
                module, output, "output", scheme.output_activations
            )

        return output

    # bind wrapped forward to module class so reference to `self` is correct
    bound_wrapped_forward = wrapped_forward.__get__(module, module.__class__)
    # set forward to wrapped forward
    setattr(f, "forward", bound_wrapped_forward)


def _maybe_calibrate_or_quantize(
    module: Module, value: Module, base_name: str, args: "QuantizationArgs"
) -> torch.Tensor:
    # only run quantized for the included stages
    if module.quantization_status not in {QuantizationStatus.CALIBRATION}:
        return value

    scale = getattr(module, f"{base_name}_scale")
    zero_point = getattr(module, f"{base_name}").data = zero_point

    if module.quantization_status == QuantizationStatus.CALIBRATION:
        # get observer and get new quant params from observation
        observer = getattr(module, f"{base_name}_observer")
        updated_scale, updated_zero_point = observer(value)

        # update scale and zero point
        scale.data = updated_scale
        zero_point.data = updated_zero_point

    return fake_quantize(value, scale, zero_point)
