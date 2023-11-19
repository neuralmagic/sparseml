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

from typing import Dict

import numpy
import onnx
from onnx import numpy_helper

import torch


__all__ = [
    "onnx_torch_matcher",
]


OP_TYPES = ["Conv", "MatMul", "Gemm", "MatMulInteger", "ConvInteger"]
QUANTIZED_LINEAR_OP_TYPES = ["QLinearConv", "QLinearMatMul"]


def onnx_torch_matcher(
    onnx_model_path: str, torch_model_path: str, 
    # epsilon: float = 1e-5
    epsilon: float = 2e-1

) -> Dict[str, str]:
    """
    [NOTE]: Macher works with dense models, may have trouble with optimized models

    Match the onnx init name to torch names as a dictionary. Dict keys
    will be one of Conv, MatMul, Gemm, MatMulInteger, ConvInteger,
    QLinearConv and QLinearMatMul.

    Layer matching based on the abs max array difference within +/- eplison

    :param onnx_model_path: path to .onnx
    :param torch_model_path: path to .pth
    """

    onnx_model = onnx.load(onnx_model_path)

    onnx_weight_names = [
        node.input[1] for node in onnx_model.graph.node if node.op_type in OP_TYPES
    ]
    onnx_weight_names.extend(
        [
            node.input[3]
            for node in onnx_model.graph.node
            if node.op_type in QUANTIZED_LINEAR_OP_TYPES
        ]
    )


    torch_model = torch.load(torch_model_path, map_location=torch.device("cpu"))

    if "state_dict" in torch_model:
        torch_model = torch_model["state_dict"]

    onnx_torch_mapper = {}
    for init in onnx_model.graph.initializer:
        if init.name in onnx_weight_names:
            arr_onnx = numpy_helper.to_array(init)

            candidates = {}
            for key, val in torch_model.items():
                arr_torch = val.numpy()
                # print(numpy.shape(arr_onnx),numpy.shape(arr_torch) )
                if numpy.shape(arr_onnx) == numpy.shape(arr_torch): 
                    diff = _mse(arr_onnx, arr_torch)
                    candidates[diff] = key

            while candidates:
                min_diff = min(candidates)
                # breakpoint()

                if min_diff > epsilon:
                    candidates = {}
                    break
                if candidates[min_diff] in onnx_torch_mapper.keys():
                    del candidates[min_diff]
                else:
                    onnx_torch_mapper[init.name] = candidates[min_diff]
                    break

    return onnx_torch_mapper


def _mse(arr_onnx, arr_torch):
    # diff = (arr_onnx - arr_torch)
    # abs_diff = numpy.sum(diff ** 2)

    # return numpy.ceil(abs_diff)

    # print(numpy.max(numpy.abs(arr_onnx - arr_torch)))
    print(numpy.min(numpy.abs(arr_onnx - arr_torch)))

    # return numpy.max(numpy.abs(arr_onnx - arr_torch))
    return numpy.min(numpy.abs(arr_onnx - arr_torch))



torch_path = f"/home/ubuntu/george/nm/sparseml/scratch/pytorch_model.bin"
onnx_path = "/home/ubuntu/george/nm/sparseml/scratch/model.onnx"

# torch_path = f"/home/ubuntu/george/nm/sparseml/scratch/model1.pth"
# onnx_path = "/home/ubuntu/george/nm/sparseml/scratch/model1.onnx"

m = onnx_torch_matcher(
    onnx_path,
    torch_path,
    100000000000,
)

print(m)
    