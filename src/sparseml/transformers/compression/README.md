# Save/Load Compressed SafeTensors

## Motivation

* Reduce disk space by saving in a compressed format for sparse models. Models in this compressed format will be loaded by vLLM for more efficient inference
* Set up the save/load architecture such that we can easily expand to additional compression formats in the future. The config should be human readable so users can understand the compression format at a quick glance

## SafeTensors File Format

For each parameter in the uncompressed state_dict, we store the following attributes 
needed for decompression in the compressed state_dict:

* compressed tensor
* bitmask
* uncompressed shape
* row offsets

```python
# dense
{
    PARAM_NAME: uncompressed_tensor
}

# compressed
{
    PARAM_NAME.compressed: compressed_tensor # 1d tensor
    PARAM_NAME.bitmask: value # 2d bitmask tensor (nrows x (ncols / 8))
Satrat marked this conversation as resolved.
    PARAM_NAME.shape: value # uncompressed shape tensor
    PARAM_NAME.row_offsets: value # 1d offsets tensor
}
```

## Example Code

```python
from sparseml.transformers import SparseAutoModelForCausalLM
from sparseml.transformers.compression import BitmaskConfig, BitmaskCompressor
from safetensors import safe_open
import os

MODEL_PATH = "zoo:llama2-7b-gsm8k_llama2_pretrain-pruned50.oneshot"
OUTPUT_PATH = "./test_compress_output"

model = SparseAutoModelForCausalLM.from_pretrained(MODEL_PATH)

sparsity_config = BitmaskConfig()
compressor = BitmaskCompressor(config=sparsity_config)

model_state_dict = model.state_dict()
sparse_state_dict = compressor.compress(model_state_dict)


model.save_pretrained(OUTPUT_PATH, safe_serialization=True, state_dict=sparse_state_dict)

safetensors_path = os.path.join(OUTPUT_PATH, "model-00001-of-00002.safetensors")
with safe_open(safetensors_path, framework="pt", device=0) as f:
    test_name = "model.layers.4.self_attn.k_proj.weight"
    bitmask = f.get_tensor(test_name + ".bitmask")
    shape = f.get_tensor(test_name + ".shape")
    values = f.get_tensor(test_name + ".compressed")
    row_offsets = f.get_tensor(test_name + ".row_offsets")
    print(f"bitmask: {bitmask}")
    print(f"shape: {shape}")
    print(f"values: {values}")
    print(f"row offsets: {row_offsets}")
```