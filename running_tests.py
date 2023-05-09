import copy

import onnx
from onnx.utils import extract_model
from collections import defaultdict
from sparseml.exporters.transforms import AddKeyValueCache
from transformers import AutoTokenizer
import onnxruntime as rt
import numpy as np

DEBUG_NODE = "onnx::Gather_392"
DEBUG_NODE_BASELINE = "onnx::Gather_459"


def add_debug_node(model_path: str, new_model_path: str, intermediate_tensor_name:str):
    """
    Add a debug node to the model to inspect the intermediate tensor
    """
    model = onnx.load(model_path)

    debug = onnx.helper.make_tensor_value_info(
        intermediate_tensor_name,
        onnx.TensorProto.INT64,
        # hacky
        shape= [None, None, None, None],
    )
    model.graph.output.extend([debug])
    onnx.save(model, new_model_path)

    # input_names = [x.name for x in model.graph.input]
    # output_names = [intermediate_tensor_name]
    # extract_model(model_path, new_model_path, input_names, output_names)
    # model = onnx.load(new_model_path)
    # onnx.save(model, new_model_path, save_as_external_data=True, all_tensors_to_one_file=True)

#### EDIT THE EXPORTED MODEL ####
"""
To obtain the `deployment/model.onnx run transformers export with the following arguments:
--task codegen --model_path codegen-350M-multi --sequence_length 1
(codegen-350-multi directory is obtained by cloning the original HF repository:
git clone https://huggingface.co/Salesforce/codegen-350M-multi
)
"""
onnx_model = onnx.load("deployment/model.onnx")
onnx_model = AddKeyValueCache().transform(onnx_model)
onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, "small_codegen.onnx", save_as_external_data=True, all_tensors_to_one_file=True)

#### RUN ONNX MODEL ####
# Setup the tested model

model_path = "small_codegen.onnx"
model_path_debug = "small_codegen_debug.onnx"

add_debug_node(model_path, model_path_debug, DEBUG_NODE)
sess = rt.InferenceSession(model_path_debug)
output_names = [out.name for out in onnx.load(model_path_debug).graph.output]

# Setup the baseline model (optimum exported)
"""
To obtain the `decoder_with_past_model.onnx` run optimum export:
optimum-cli export onnx --model Salesforce/codegen-350M-multi codegen-350M-multi
"""
baseline_model_path = "/home/ubuntu/damian/deepsparse/codegen-350M-multi/decoder_with_past_model.onnx"
baseline_model_path_debug = "debug.onnx"
add_debug_node(baseline_model_path, baseline_model_path_debug, DEBUG_NODE_BASELINE)
sess_baseline = rt.InferenceSession(baseline_model_path_debug)
output_names_baseline = [out.name for out in onnx.load(baseline_model_path_debug).graph.output]

# Setup the input
input_sequence = "def hello_world():"
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi")
model_input = tokenizer(input_sequence, return_tensors='np').data
input_ids = model_input['input_ids']

# Setup the cache
batch_size = 1
head_dim = 64
num_heads = 16
past_length = 0
seq_length = 1


# Setup the cache (tested and baseline)
kv_cache = defaultdict(np.ndarray)
kv_cache_baseline = defaultdict(np.ndarray)

for i in range(20):
    kv_cache[f"past_key_values.{i}.value"] = np.zeros((batch_size, num_heads, past_length, head_dim)).astype(np.float32)
    kv_cache[f"past_key_values.{i}.key"] = np.zeros((batch_size, num_heads, head_dim, past_length)).astype(np.float32)

    kv_cache_baseline[f"past_key_values.{i}.value"] = np.zeros((batch_size, num_heads, past_length, head_dim)).astype(np.float32)
    kv_cache_baseline[f"past_key_values.{i}.key"] = np.zeros((batch_size, num_heads, past_length, head_dim)).astype(np.float32)


for i, token in enumerate(input_ids[0]):
    attention_mask = np.ones((batch_size, seq_length + past_length)).astype(np.int64)
    input_ids = np.array([[token]])

    # Inference tested model
    out = sess.run(None, {"input_ids": input_ids,"attention_mask": attention_mask, **kv_cache})
    *kv_cache, logits, debug_node = out
    new_kv_cache = {k.replace("present",  "past_key_values"): v for k, v in zip(output_names[:-1], kv_cache)}

    # Inference baseline model
    out_baseline = sess_baseline.run(None, {"input_ids": input_ids,
                                            "attention_mask": attention_mask,
                                            **kv_cache_baseline})
    logits_baseline, *kv_cache_baseline, debug_node_baseline = out_baseline
    new_kv_cache_baseline = {k.replace("present", "past_key_values"): v for k, v in zip(output_names_baseline[1:], kv_cache_baseline)}

    matched_keys = 0
    matched_values = 0
    for k,v in new_kv_cache_baseline.items():
        if k in new_kv_cache:
            if k.endswith("key"):
                if np.allclose(v.transpose(0,1,3,2), new_kv_cache[k], atol=1e-3):
                    matched_keys +=1
            else:
                if np.allclose(v, new_kv_cache[k], atol=1e-3):
                    matched_values +=1

    print("Iteration: ", i)
    print("Matched keys: ", matched_keys, "/20")
    print("Matched values: ", matched_values, "/20")
    print("Matched logits: ", np.allclose(logits, logits_baseline, atol=1e-3))
    print(f"Predicted tokens - {np.argmax(logits[0,0,:])} (baseline {np.argmax(logits_baseline[0,0,:])})")
    kv_cache_baseline = new_kv_cache_baseline
    kv_cache = new_kv_cache
    past_length +=1
    token = np.argmax(logits[0,0,:])




