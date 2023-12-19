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

from sparseml.exporters.transforms import OnnxTransform
from sparseml.exporters.transforms.utils import (
    INITIALIZER_MATCH,
    get_quantization_params,
    get_structural_matches,
    optional_node,
)
from sparseml.exporters.transforms.utils.helpers import quantize_array
from sparseml.onnx.utils import ONNXGraph


__all__ = ["WeightsOnly"]


class WeightsOnly(OnnxTransform):
    """
    A transform that converts floating point weights to INT8.

    Transforms
    ```
    |   weights (initializer)
    |     |
    |     Q
    |     |
    |     Dq
    |     |
    |  Transpose (optional)
    ```
    (where `Q` is QuantizeLinear, and `Dq` is DequantizeLinear)

    into

    ```
    |   weights (INT8 initializer)
    |      |
    |      Dq
    ```
    """

    def transform(self, model: ModelProto) -> ModelProto:
        graph = ONNXGraph(model)
        matches = get_structural_matches(
            graph,
            parent_ops=[[INITIALIZER_MATCH, "QuantizeLinear"]],
            op_type="DequantizeLinear",
            children_ops=[[optional_node("Transpose")]],
        )

        for match in matches:
            self.log_match(match)

            initializer, qnode = match.parents[0]
            dqnode = match.node
            transpose_node = match.children[0][0]

            quantize_params = get_quantization_params(model, qnode, include_target=True)

            quantized_array = quantize_array(
                quantize_params.target,
                quantize_params.scale,
                quantize_params.zero_point,
                quantize_params.zero_point.dtype,
            )

            if transpose_node is not None:
                # Transpose array
                quantized_array = quantized_array.transpose()

                # Move the output of Dequantize node to the
                # output of the Transpose node (to be removed)
                # such that the output of the block remains the same
                dqnode.output[0] = transpose_node.output[0]

                # Find which axis to apply dequantization scale.
                # This will be the axis being permuted by the
                # Transpose node
                for attribute in transpose_node.attribute:
                    if attribute.name == "perm":
                        permutation_axis = attribute.ints
                        break

                for attribute in dqnode.attribute:
                    if attribute.name == "axis":
                        original_axis = attribute.i
                        permutation_axis.remove(original_axis)
                        new_axis = permutation_axis[0]
                        attribute.i = new_axis
                        break

            quantized_initializer_name = initializer.name + "_quantized"
            quantized_initializer = numpy_helper.from_array(
                quantized_array, name=quantized_initializer_name
            )
            model.graph.initializer.append(quantized_initializer)

            dqnode.input[0] = quantized_initializer_name

            # Clean up
            self.delete_node_deferred(qnode)

            if transpose_node is not None:
                self.delete_node_deferred(transpose_node)

        return model
