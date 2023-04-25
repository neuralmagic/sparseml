import onnx
from collections import defaultdict
from sparseml.exporters.transforms import AddKeyValueCache

import onnxruntime as rt
import numpy as np

#### EDIT THE EXPORTED MODEL ####
onnx_model = onnx.load("deployment/model.onnx")
onnx_model = AddKeyValueCache().transform(onnx_model)
onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, "small_codegen.onnx", save_as_external_data=True, all_tensors_to_one_file=True)

#### RUN ONNX MODEL ####

sess = rt.InferenceSession("small_codegen.onnx")


batch_size = 1
head_dim = 64
num_heads = 16
past_length = 2
seq_length = 1

input_ids = np.random.randint(0, 100, (batch_size, seq_length)).astype(np.int64)
kv_cache = defaultdict(np.ndarray)
for i in range(20):
    kv_cache[f"past_value_{i}"] = np.random.randn(batch_size, num_heads, past_length, head_dim).astype(np.float32)
    kv_cache[f"past_key_{i}"] = np.random.randn(batch_size, num_heads, head_dim, past_length).astype(np.float32)

out = sess.run(None, {"input_ids": input_ids, **kv_cache})

#### PYTORCH SANITY CHECK ####
# from transformers import AutoTokenizer, AutoModelForCausalLM
#
# tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi")
# model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-multi")
#
# text = "def hello_world():"
# input_ids = tokenizer(text, return_tensors="pt").input_ids
#
# generated_ids = model.generate(input_ids, max_length=128)
