from onnx import ModelProto, numpy_helper, helper
from sparseml.onnx.utils import ONNXGraph
from sparseml.exporters.transforms import OnnxTransform
from sparseml.exporters.transforms.utils import get_structural_matches, INITIALIZER_MATCH, quantize_array, delete_quant_node, assert_node_type
import logging
_LOGGER = logging.getLogger(__name__)

def quantize_embedding(match, model):
    graph = ONNXGraph(model)

    input_quantize_node = match.parents[1][1]
    input_dequantize_node = match.parents[1][2]

    # quantize embedding
    embedding_initializer = graph.get_init_by_name(input_quantize_node.input[0])
    scale_initializer = graph.get_init_by_name(input_quantize_node.input[1])
    zero_point_initializer = graph.get_init_by_name(input_quantize_node.input[2])

    # arrays from embedding initializer
    embedding = numpy_helper.to_array(embedding_initializer)
    scale = numpy_helper.to_array(scale_initializer)
    zero_point = numpy_helper.to_array(zero_point_initializer)

    embedding_quant = quantize_array(embedding, scale, zero_point, zero_point.dtype)
    embedding_quant_initializer = numpy_helper.from_array(embedding_quant, name=f"{embedding_initializer.name}_quant")

    # update graph
    model.graph.initializer.append(embedding_quant_initializer)
    match.node.input[0] = embedding_quant_initializer.name

    # detect QDQ block on output
    qdq_output = False

    output_quant_node = match.children

    if not output_quant_node:
        embedding_dequant_input = f"{match.node.output[0]}_quant"
        embedding_dequant_output = match.node.output[0]
        dequantize_linear_node = helper.make_node(
            "DequantizeLinear",
            [embedding_dequant_input , input_dequantize_node.inputs[1], input_dequantize_node.inputs[2]],
            [embedding_dequant_output],
            name="dequantize_linear_node_0",
        )
        pass

    # if output_quant_node:
    #     if assert_node_type(output_quant_node, "QuantizeLinear"):
    #         output_dequant_node = graph.get_node_single_child(output_quant_node)
    #         qdq_output = assert_node_type(output_quant_node, "DequantizeLinear")
    #
    # if qdq_output:
    #     # forward gather output to dequant input
    #     output_dequant_node.input[0] = match.node.output[0]
    #     output_dequant_node.input[1] = input_quantize_node.input[1]
    #     output_dequant_node.input[2] = input_quantize_node.input[2]
    #     # delete unnecessary quantize and dequantize ops
    #     delete_quant_node(model, input_quantize_node)
    #     delete_quant_node(model, input_dequantize_node)
    #     delete_quant_node(model, output_quant_node)
    #
    # else:
    #     # use input dequant to dequantize output
    #     embedding_quant_output_id = f"{match.node.output[0]}_quant"
    #     input_dequantize_node.input[0] = embedding_quant_output_id
    #     input_dequantize_node.output[0] = match.node.output[0]
    #     match.node.output[0] = embedding_quant_output_id
    #     delete_quant_node(model, input_quantize_node)

    graph.update()
    graph.delete_unused_initializers()

    return model


class QuantizeQATEmbedding(OnnxTransform):
    """
    A transformation for quantizing
    qat embeddings

    Starting with:
    |    INPUT    QuantizeLinear (with constant embedding)
    |      |          |
    |      |     DequantizeLinear
    |      |         |
    |         Gather
    |           |
    |       QuantizeLinear (Optional)
    |           |
    |       DequantizeLinear (Optional)
    |           |
    |         OUTPUT

    Converts to:
    |   INPUT
    |     |
    |   Gather(UINT8 data initializer)
    |     |
    |   DequantizeLinear
    |     |
    |   OUTPUT
    """

    def transform(self, model: ModelProto) -> ModelProto:
        count_converted_nodes = 0
        graph = ONNXGraph(model)
        for match in get_structural_matches(
                graph,
                parent_ops=[
                    [],
                    [
                        INITIALIZER_MATCH,
                        "QuantizeLinear",
                        "DequantizeLinear",
                    ],
                    ],
                op_type="Gather",
        ):
            _LOGGER.debug(
                f"Matched quantizable Conv weight and bias: {match.node.name}"
            )
            model = quantize_embedding(match, model)

            count_converted_nodes += 1

        if count_converted_nodes > 0:
            _LOGGER.info(f"Converted {count_converted_nodes} QAT embedding ops to UINT8")
        return model