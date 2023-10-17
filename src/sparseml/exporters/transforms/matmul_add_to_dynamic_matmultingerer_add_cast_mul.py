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

import onnx
from onnx import ModelProto, numpy_helper

from sparseml.exporters.transforms import OnnxTransform
from sparseml.exporters.transforms.utils import (
    INITIALIZER_MATCH,
    MatchResult,
    get_quantization_params,
    get_structural_matches,
    optional_node,
    quantize_array,
)
from sparseml.onnx.utils import ONNXGraph


__all__ = ["MatMulAddToDynamicMatMulIntegerAddCastMul"]


def find_input_to_cast(graph, cast_node):
    parent_nodes = graph.get_node_parents(cast_node)
    if len(parent_nodes) == 1 and parent_nodes[0].op_type == "Cast":
        return find_input_to_cast(graph, parent_nodes[0])
    else:
        return cast_node.input[0]


def find_zero_point(graph, zero_point_add_node, scale_div_node):
    # Check if zero point goes through Cast node before being added
    for parent in graph.get_node_parents(zero_point_add_node):
        if parent == scale_div_node:
            continue

        if parent.op_type == "Cast":
            return find_input_to_cast(graph, parent)

    # If here there is no Cast node that is a parent to
    # zero point add.
    # This means the data type is not correct.
    raise Exception(
        f"At least one parent to {zero_point_add_node.name} must be a Cast node"
    )


class MatMulAddToDynamicMatMulIntegerAddCastMul(OnnxTransform):
    """
    A transform that attempts, if possible, to convert MatMul nodes into
    their quantized representation.
    This MatMul is the result of quantizing native torch.matmul using QATMatMul

    NOTE: we expect that INPUT_0 and INPUT_1 are not initializers (in case where
    both inputs are activations)


    Transforms
    ```
    |   scale   input   zero_point
    |       |    |       |
    |       |    |     Cast (optional)
    |       |    |       |
    |       |   Div    Cast (optional)
    |       |     |   |
    |       |      Add
    |       |       |
    |       |     Round     weight (initializer)
    |       |       |          |
    |       |     Clip         Q
    |       |       |          |
    |       |      Sub        DQ
    |       |    |            |
    |         Mul          Transpose (optional)
    |            |           |
    |               MatMul
    |                  |
    |                 Add (constant bias) (optional)
    ```

    into

    ```
    |   input   scale   zero_point
    |        |    |    |
    |             Q
    |             |
    |         MatMulInteger (with constant uint8 kernel)
    |             |
    |            Cast (INT32 -> FP32)
    |             |
    |            Mul (Rescale from bias scale)
    |             |
    |            Add (constant bias) (optional)
    ```
    """

    def transform(self, model: ModelProto) -> ModelProto:
        graph = ONNXGraph(model)

        parent_ops = [
            [
                "Div",
                "Add",
                "Round",
                "Clip",
                "Sub",
                "Mul",
            ],
            [
                # weight should be initializer
                INITIALIZER_MATCH,
                "QuantizeLinear",
                "DequantizeLinear",
                optional_node("Transpose"),
            ],
        ]

        matches = []
        for order in range(2):
            matches_ = get_structural_matches(
                graph,
                op_type="MatMul",
                parent_ops=parent_ops,
                children_ops=[[optional_node("Add")]],
            )
            if order == 0:
                # reverse the parent order (weights on first input)
                # before second pass
                parent_ops.reverse()
            else:
                # if matched using reverse order, need to
                # revert parents order to match transformation logic
                for m in matches_:
                    m.parents.reverse()
            matches.extend(matches_)

        processed_div_nodes = []
        for match in matches:
            scale_div_node = match.parents[0][0]
            if scale_div_node in processed_div_nodes:
                new_quantization = False
            else:
                new_quantization = True
                processed_div_nodes.append(scale_div_node)
            self.log_match(match)
            self._transform_match(model, graph, match, new_quantization)
        return model

    def _transform_match(
        self,
        model: ModelProto,
        graph: ONNXGraph,
        match: MatchResult,
        new_quantization: bool,
    ):
        (
            scale_div,
            zero_point_add,
            round_node,
            clip,
            sub_node,
            scale_mul,
        ) = match.parents[0]
        weight_init, weight_quant, weight_dequant, opt_transpose = match.parents[1]

        # Find scale and input tensor
        # Scale is the only input in common between the div and the mul nodes
        for inp in scale_div.input:
            if inp in scale_mul.input:
                scale = inp
            else:
                input_tensor = inp

        # Find zero point
        zero_point = find_zero_point(graph, zero_point_add, scale_div)

        # Find weight quantization parameters
        weight_quantize_params = get_quantization_params(
            model, weight_quant, include_target=True
        )
        # sanity check - matching handles this
        assert weight_quantize_params.target is not None

        # Quantize weights and add to graph
        quantized_weight = quantize_array(
            weight_quantize_params.target,
            weight_quantize_params.scale,
            weight_quantize_params.zero_point,
            weight_quantize_params.zero_point.dtype,
        )
        if opt_transpose is not None:
            quantized_weight = quantized_weight.transpose()
        quantized_weight_name = f"{match.node.name}.weight_quantized"
        quantized_weight_initializer = numpy_helper.from_array(
            quantized_weight, name=quantized_weight_name
        )
        model.graph.initializer.append(quantized_weight_initializer)

        weight_scale_name = f"{match.node.name}.weight_scale"
        weight_scale_initializer = numpy_helper.from_array(
            weight_quantize_params.scale,
            name=weight_scale_name,
        )
        model.graph.initializer.append(weight_scale_initializer)

        if new_quantization:
            # Create QuantizeLinear node
            quant_node = onnx.helper.make_node(
                "QuantizeLinear",
                [input_tensor, scale, zero_point],
                [f"{input_tensor}_input_quant"],
                f"{input_tensor}_input_quant",
            )
            self.add_node_deferred(quant_node)

        # Create MatMulInteger node
        matmul_node = onnx.helper.make_node(
            "MatMulInteger",
            [f"{input_tensor}_input_quant", quantized_weight_name, zero_point],
            [f"{match.node.name}_output"],
            f"{match.node.name}_quant",
        )
        self.add_node_deferred(matmul_node)

        # Create Cast node
        cast_node = onnx.helper.make_node(
            "Cast",
            matmul_node.output,
            [f"{match.node.name}_output_cast"],
            f"{match.node.name}_output_cast",
            to=getattr(onnx.TensorProto, "FLOAT"),  # get Float32 enum id
        )
        self.add_node_deferred(cast_node)

        # Create Mul node (weight scale)
        weight_scale_node = onnx.helper.make_node(
            "Mul",
            [f"{match.node.name}_output_cast", weight_scale_name],
            [f"{match.node.name}_output_weight_scale"],
            f"{match.node.name}_weight_scale",
        )
        self.add_node_deferred(weight_scale_node)

        # Create Mul node (activation scale)
        activation_scale_node = onnx.helper.make_node(
            "Mul",
            [f"{match.node.name}_output_weight_scale", scale],
            match.node.output,
            f"{match.node.name}_activation_scale",
        )
        self.add_node_deferred(activation_scale_node)

        # Clean up
        if new_quantization:
            self.delete_node_deferred(scale_div)
            self.delete_node_deferred(zero_point_add)
            self.delete_node_deferred(round_node)
            self.delete_node_deferred(clip)
            self.delete_node_deferred(sub_node)
            self.delete_node_deferred(scale_mul)

        self.delete_node_deferred(weight_quant)
        self.delete_node_deferred(weight_dequant)
        if opt_transpose is not None:
            self.delete_node_deferred(opt_transpose)
        self.delete_node_deferred(match.node)
