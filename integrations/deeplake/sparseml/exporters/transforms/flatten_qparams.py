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

import numpy
from onnx import ModelProto, numpy_helper

from sparseml.exporters.transforms.onnx_transform import OnnxTransform
from sparseml.onnx.utils.graph_editor import ONNXGraph


__all__ = ["FlattenQParams"]


class FlattenQParams(OnnxTransform):
    """
    Transforms any QuantizeLinear/DequantizeLinear that have
    zero_point/scale with shapes `(1,)` into shape `()`
    """

    def transform(self, model: ModelProto) -> ModelProto:
        graph = ONNXGraph(model)

        inits_to_flatten = set()

        for node in model.graph.node:
            if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
                # scale is required if the input is an initializer
                scale_init = graph.get_init_by_name(node.input[1])
                if scale_init is not None and list(scale_init.dims) == [1]:
                    inits_to_flatten.add(node.input[1])

                    # zero_point is optional AND shape must match
                    # scale. so if scale is (1,), then so will zero point
                    if len(node.input) == 3:
                        inits_to_flatten.add(node.input[2])

        for init in model.graph.initializer:
            if init.name not in inits_to_flatten:
                continue
            self.log_match(init)
            a = numpy_helper.to_array(init)
            assert a.shape == (1,)
            b = numpy.array(a[0])
            assert b.shape == ()
            assert b.dtype == a.dtype
            init.CopyFrom(numpy_helper.from_array(b, name=init.name))

        return model
