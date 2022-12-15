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

from typing import Any

import numpy
import torch


def quantize_array_torch(
    array: numpy.ndarray, scale: float, zero_point: int, dtype: Any
) -> numpy.ndarray:

    if dtype == numpy.uint8:
        tensor_dtype = torch.quint8
    elif dtype == numpy.int8:
        tensor_dtype = torch.qint8
    elif dtype == numpy.int32:
        tensor_dtype = torch.qint32

    tensor = torch.Tensor(array.copy()).to(torch.float32)
    if isinstance(scale, numpy.ndarray):
        scale = scale.item()
    if isinstance(zero_point, numpy.ndarray):
        zero_point = zero_point.item()

    quant_tensor = torch.quantize_per_tensor(tensor, scale, zero_point, tensor_dtype)
    return quant_tensor.int_repr().numpy()
