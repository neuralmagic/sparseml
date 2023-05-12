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


from abc import abstractmethod

import onnx
from onnx import ModelProto

from sparseml.exporters.transforms.onnx_transform import OnnxTransform


__all__ = ["CacheLengthAdjustment"]


class CacheLengthAdjustment(OnnxTransform):
    """
    Base class for model architecture specific transforms to adjust graph
    based on a cache length input
    """

    CACHE_LENGTH_NAME = "cache_length"

    @abstractmethod
    def update_model_for_cache_length(self, model: ModelProto) -> ModelProto:
        """
        updates the model to handle cache length after a cache_length
        graph input has been added

        :param model: model to update
        :return: updated model
        """
        raise NotImplementedError

    @classmethod
    def add_cache_length_input(cls, model: ModelProto) -> ModelProto:
        """
        adds cache length as an input to the model

        :param model: model to update
        :return: updated model
        """
        cache_length_input = onnx.helper.make_tensor_value_info(
            cls.CACHE_LENGTH_NAME,
            onnx.TensorProto.INT64,
            [1],
        )
        model.graph.input.append(cache_length_input)
        return model

    def transform(self, model: ModelProto) -> ModelProto:
        # add cache_length as graph input
        model = self.add_cache_length_input(model)
        # abc method to propagate cache_length where necessary
        model = self.update_model_for_cache_length(model)
        return model
