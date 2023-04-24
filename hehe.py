import onnx
from onnx.external_data_helper import load_external_data_for_model
from sparseml.exporters.transforms import AddKeyValueCache
import onnxruntime as rt
import numpy as np

# onnx_model = onnx.load("opt-350m/deployment/small_opt.onnx")
# load_external_data_for_model(onnx_model,"opt-350m/deployment/filename")
# onnx_model = AddKeyValueCache().transform(onnx_model)
#
# onnx.checker.check_model(onnx_model)
# onnx.save(onnx_model, "small_opt_formatted.onnx", save_as_external_data=True, all_tensors_to_one_file=True)

sess = rt.InferenceSession("small_opt_formatted.onnx")
label_name = "Transpose_664"
input = {"input_ids" : np.random.randint(0, 100, (1, 384))}
input["attention_mask"] = np.ones((1, 384)).astype(np.int)
for i in range(24):
    input[f"past_value_{i}"] = np.random.randn(1,32,1,1).astype(np.float32)
    input[f"past_key_{i}"] = np.random.randn(32, 1, 1).astype(np.float32)
sess.run([label_name], {**input}, )