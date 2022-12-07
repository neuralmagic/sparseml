from onnx import ModelProto, numpy_helper
from typing import Union
from sparseml.onnx.utils import ONNXGraph, check_load_model, validate_onnx_file
from sparseml.exporters.transforms import BaseTransform
from sparseml.exporters.transforms.utils.helpers import INITIALIZER_MATCH, quantize_array
from sparseml.exporters.transforms.utils import iter_structural_matches
import logging
_LOGGER = logging.getLogger(__name__)

def quantize_embedding(model, match):

    graph = ONNXGraph(model.graph)
    input_quantize_node = match.parents[1][1]
    input_dequantize_node = match.parents[1][2]

    # quantize embedding
    embedding_initializer = graph.get_init_by_name(input_quantize_node.input[0])
    scale_initializer = graph.get_init_by_name(input_quantize_node.input[1])
    zero_point_initializer = graph.get_init_by_name(input_quantize_node.input[2])

    embedding = numpy_helper.to_array(embedding_initializer)
    scale = numpy_helper.to_array(scale_initializer)
    zero_point = numpy_helper.to_array(zero_point_initializer)

    embedding_quant = quantize_array(embedding, scale, zero_point, zero_point.dtype)
    embedding_quant_initializer = numpy_helper.from_array(embedding_quant, name=f"{embedding_initializer.name}_quant")

    # update graph
    model.graph.initializer.append(embedding_quant_initializer)
    match.node.input[0] = embedding_quant_initializer.name

    # detect QDQ block on output
    output_quant_node = graph.get_node_single_child(match.node)
    if assert_node_type(output_quant_node, "QuantizeLinear"):
        output_dequant_node = graph.get_node_single_child(output_quant_node)
        qdq_output = assert_node_type(output_quant_node, "DequantizeLinear")
    else:
        qdq_output = False

    if qdq_output:
        # forward gather output to dequant input
        output_dequant_node.input[0] = match.node.output[0]
        output_dequant_node.input[1] = input_quantize_node.input[1]
        output_dequant_node.input[2] = input_quantize_node.input[2]
        # delete unnecessary quantize and dequantize ops
        delete_quant_node(model, input_quantize_node)
        delete_quant_node(model, input_dequantize_node)
        delete_quant_node(model, output_quant_node)

    else:
        # use input dequant to dequantize output
        embedding_quant_output_id = f"{match.node.output[0]}_quant"
        input_dequant_node.input[0] = embedding_quant_output_id
        input_dequant_node.output[0] = match.node.output[0]
        match.node.output[0] = embedding_quant_output_id

        delete_quant_node(model, input_quantize_node)
    graph.update()
    converted_nodes += 1

graph.delete_unused_initializers()




class QuantizeQATEmbedding(BaseTransform):
    """
    Folds any `Identity` initializer node. Such a node is defined by:
     - having a single input
     - having a single output
     - being an `Identity` operation
     - being an `initializer` node
    """

    def _transform(self, model: ModelProto) -> ModelProto:
        graph = ONNXGraph(model)
        matches = []

        count_converted_nodes = 0
        graph = ONNXGraph(model)
        for match in iter_structural_matches(
                graph,
                parent_ops=[
                    [],
                    [# weight should be initializer
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
            model = quantize_embedding(model, match)

            count_converted_nodes += 1

        if count_converted_nodes > 0:
            _LOGGER.info(f"Converted {converted_nodes} QAT embedding ops to UINT8")
        return model

    def _validate_input(self, model: ModelProto):
        validate_onnx_file(model)

    def _validate_output(self, model: ModelProto):
        validate_onnx_file(model)

    def apply(self, model: Union[ModelProto, str]) -> ModelProto:
        onnx_model = check_load_model(model)
        self._validate_input(onnx_model)
        onnx_model = self._transform(onnx_model)
        self._validate_output(onnx_model)
        return onnx_model