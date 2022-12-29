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

from onnx import ModelProto

from sparseml.exporters.transforms.onnx_transform import OnnxTransform


__all__ = ["UnwrapBatchNorms"]


class UnwrapBatchNorms(OnnxTransform):
    """
    Removes `.bn_wrapper_replace_me` from all initializers and nodes
    """

    def transform(self, model: ModelProto) -> ModelProto:
        for init in model.graph.initializer:
            init.name = init.name.replace(".bn_wrapper_replace_me", "")
        for node in model.graph.node:
            for i in range(len(node.input)):
                node.input[i] = node.input[i].replace(".bn_wrapper_replace_me", "")
            for i in range(len(node.output)):
                node.output[i] = node.output[i].replace(".bn_wrapper_replace_me", "")
        return model
