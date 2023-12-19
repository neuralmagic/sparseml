from onnx import ModelProto, numpy_helper

from sparseml.exporters.transforms import OnnxTransform
from sparseml.onnx.utils import ONNXGraph
from sparseml.exporters.transforms.utils import (
    INITIALIZER_MATCH,
    get_structural_matches,
    get_quantization_params,
    optional_node,
)
from sparseml.exporters.transforms.utils.helpers import quantize_array

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
            parent_ops=[
                [INITIALIZER_MATCH, "QuantizeLinear"]
            ],
            op_type="DequantizeLinear",
            children_ops=[[optional_node("Transpose")]],
        )

        for match in matches:
            self.log_match(match)

            initializer, qnode = match.parents[0]
            dqnode = match.node
            transpose_node = match.children[0][0]

            quantize_params = get_quantization_params(
                model, qnode, include_target=True
            )

            quantized_array = quantize_array(
                quantize_params.target,
                quantize_params.scale,
                quantize_params.zero_point,
                quantize_params.zero_point.dtype,
            )

            if transpose_node is not None:
                quantized_array = quantized_array.transpose()
                dqnode.output[0] = transpose_node.output[0]

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