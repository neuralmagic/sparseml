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

MODEL_PATH = "zoo:llama2-7b-gsm8k_llama2_pretrain-pruned50.oneshot"
OUTPUT_PATH = "./test_compress_output"

model = SparseAutoModelForCausalLM.from_pretrained(MODEL_PATH)

sparsity_config = BitmaskConfig()
compressor = BitmaskCompressor(config=sparsity_config)

# compresses the model using Bitmask compression
model_state_dict = model.state_dict()
sparse_state_dict = compressor.compress(model_state_dict)

# save the compressed model
model.save_pretrained(
    OUTPUT_PATH, 
    safe_serialization=True, 
    state_dict=sparse_state_dict)

# decompress the compressed state dict
dense_state_dict = compressor.decompress(OUTPUT_PATH)

# use the dense state dict to reload the model
model_again = model = SparseAutoModelForCausalLM.from_pretrained(
    OUTPUT_PATH, 
    state_dict=dense_state_dict
)
```