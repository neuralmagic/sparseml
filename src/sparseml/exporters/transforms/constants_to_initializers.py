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

from onnx import ModelProto, numpy_helper

from sparseml.exporters.transforms.onnx_transform import OnnxTransform


__all__ = ["ConstantsToInitializers"]


class ConstantsToInitializers(OnnxTransform):
    """
    Converts any `Constant` nodes into initializers
    """

    def transform(self, model: ModelProto) -> ModelProto:
        for node in model.graph.node:
            if node.op_type != "Constant" or len(node.attribute) != 1:
                continue
            self.log_match(node)
            const_array = numpy_helper.to_array(node.attribute[0].t)
            initializer = numpy_helper.from_array(const_array, name=node.output[0])
            model.graph.initializer.append(initializer)
            self.delete_node_deferred(node)
        return model
