import onnx
from sparseml.exporters.transforms import QuantizeQATEmbedding

def _create_test_model(with_quantize_linear=False, with_dequantize_linear=False):
    """
     Creates a test model with a convolution node and quantize/dequantize nodes

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
    """

    input = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, (3,))
    embedding = onnx.helper.make_tensor("embedding", onnx.TensorProto.FLOAT, (1,), [1])

    x_scale = onnx.helper.make_tensor("x_scale", onnx.TensorProto.FLOAT, (1,), [1])
    y_scale = onnx.helper.make_tensor("y_scale", onnx.TensorProto.FLOAT, (1,), [1])
    zero_point = onnx.helper.make_tensor("zero_point", onnx.TensorProto.INT8, (1,), [1])

    model_output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, (3,)
    )
    quantize_linear_node_0 = onnx.helper.make_node(
        "QuantizeLinear",
        ["embedding", "y_scale", "zero_point"],
        ["quant_linear_0_output"],
        name="quantize_linear_node_0",
    )
    dequantize_linear_node_0 = onnx.helper.make_node(
        "DequantizeLinear",
        ["quant_linear_0_output", "x_scale", "zero_point"],
        ["dequant_linear_0_output"],
        name="dequantize_linear_node_0",
    )

    gather_node = onnx.helper.make_node(
        "Gather",
        ["input", "dequant_linear_0_output"],
        ["gather_output"],
        name = "gather_node"
    )

    graph = onnx.helper.make_graph(
        nodes=[
            quantize_linear_node_0,
            dequantize_linear_node_0,
            gather_node,
        ],
        name="g",
        inputs=[input],
        initializer=[x_scale, y_scale, embedding, zero_point],
        outputs=[model_output],
    )

    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model)
    return model


def test_convert_quantizable_matmul():
    model = _create_test_model()
    transform = QuantizeQATEmbedding()
    model = transform(model)
    #testing_function(model)
    onnx.checker.check_model(model)


