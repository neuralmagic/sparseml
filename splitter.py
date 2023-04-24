import onnx

# Load the ONNX model
input_path = 'opt-350m/deployment/model.onnx'
output_path = 'model_before_split.onnx'
onnx.utils.extract_model(input_path, output_path, [input.name for input in onnx.load(input_path).graph.input], ["onnx::Add_876"])
print('done')
