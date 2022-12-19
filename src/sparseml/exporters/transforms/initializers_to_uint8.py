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


__all__ = ["InitializersToUint8"]


class InitializersToUint8(OnnxTransform):
    """
    Converts any initializers with int8 dtype to uint8
    """

    def transform(self, model: ModelProto) -> ModelProto:
        initializers_to_add = []
        initializers_to_del = []
        for init in model.graph.initializer:
            arr_int8 = numpy_helper.to_array(init)
            if arr_int8.dtype != numpy.int8:
                continue
            self.log_match(init)
            arr_uint8 = (arr_int8.astype(numpy.int32) + 128).astype(numpy.uint8)
            init_uint8 = numpy_helper.from_array(arr_uint8, name=init.name)
            initializers_to_del.append(init)
            initializers_to_add.append(init_uint8)

        model.graph.initializer.extend(initializers_to_add)
        for init in initializers_to_del:
            model.graph.initializer.remove(init)
        return model
