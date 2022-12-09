from onnx import ModelProto, numpy_helper, helper
from sparseml.onnx.utils import ONNXGraph, remove_node_and_params_from_graph
from sparseml.exporters.transforms import OnnxTransform
from sparseml.exporters.transforms.utils import get_structural_matches, optional_node, INITIALIZER_MATCH, quantize_array, delete_quant_node, assert_node_type
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
    match.node.input[1] = embedding_quant_initializer.name

    # detect QDQ block on output
    children = match.children
    qdq_output = False
    if children:
        if len(children[0]) == 2:
            if assert_node_type(children[0][0], "QuantizeLinear") and assert_node_type(children[0][1], "DequantizeLinear"):
                qdq_output = True

    if qdq_output:
        output_dequantized_node = children[0][1]
        output_quantized_node = children[0][0]
        # forward gather output to dequant input
        output_dequantized_node.input[0] = match.node.output[0]
        output_dequantized_node.input[1] = input_dequantize_node.input[1]
        output_dequantized_node.input[2] = input_dequantize_node.input[2]
        # delete unnecessary quantize and dequantize ops
        remove_node_and_params_from_graph(model, input_quantize_node)
        remove_node_and_params_from_graph(model, input_dequantize_node)
        remove_node_and_params_from_graph(model, output_quantized_node)

    else:
        # add new dequantize node after the match node
        match_node_quant_output = f"{match.node.output[0]}_quant"

        new_dequantize_node = helper.make_node(
            "DequantizeLinear",
            inputs=[match_node_quant_output, input_dequantize_node.input[1], input_dequantize_node.input[2]],
            outputs=[match.node.output[0]],
            name=f"dequantize_linear_{match.node.name}",
        )
        model.graph.node.append(new_dequantize_node)
        match.node.output[0] = match_node_quant_output

        remove_node_and_params_from_graph(model, input_quantize_node)
        remove_node_and_params_from_graph(model, input_dequantize_node)

    #graph.update()
    #graph.delete_unused_initializers()
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
                children_ops=[[optional_node("QuantizeLinear"),optional_node("DequantizeLinear")]],
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