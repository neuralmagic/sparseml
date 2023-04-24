import onnx
from transformers import pipeline
import onnxruntime as rt
import numpy as np
from collections import defaultdict
from sparseml.exporters.transforms import AddKeyValueCache

onnx_model = onnx.load("deployment/model.onnx")
onnx_model = AddKeyValueCache().transform(onnx_model)

onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, "small_opt_formatted.onnx", save_as_external_data=True, all_tensors_to_one_file=True)

batch_size = 1
head_dim = 64
num_heads = 16
past_length = 2
seq_length = 1
kv_cache = defaultdict(np.ndarray)
sess = rt.InferenceSession("small_opt_formatted.onnx")
label_name = "MatMul_665"
for i in range(24):
    kv_cache[f"past_value_{i}"] = np.random.randn(batch_size * num_heads, past_length, head_dim).astype(np.float32)
    kv_cache[f"past_key_{i}"] = np.random.randn(batch_size * num_heads, head_dim, past_length).astype(np.float32)
sess.run(None, {"input_ids": np.random.randint(0, 100, (batch_size, seq_length)),
                        "attention_mask": np.ones((batch_size, seq_length + past_length)).astype(np.int),
                        **kv_cache})

from transformers import pipeline
import onnxruntime as rt
import numpy as np



generator = pipeline('text-generation', model="facebook/opt-350m")
generator("Hello")