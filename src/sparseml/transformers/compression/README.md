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
from sparseml.utils.pytorch.utils import measure_cuda_memory
from tqdm import tqdm
import torch

MODEL_PATH = "zoo:llama2-7b-gsm8k_llama2_pretrain-pruned50.oneshot"
OUTPUT_PATH = "./test_compress_output"

torch.cuda.set_device(0)
with measure_cuda_memory() as m:
    model = SparseAutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="cuda:0")
print(f"Load dense model peak GPU {m.overall_peak_memory / float(2**30):.4f} GB")

sparsity_config = BitmaskConfig()
compressor = BitmaskCompressor(config=sparsity_config)

# compresses the model using Bitmask compression
with measure_cuda_memory() as m:
    model_state_dict = model.state_dict()
    sparse_state_dict = compressor.compress(model_state_dict)

    # save the compressed model
    model.save_pretrained(
        OUTPUT_PATH, 
        safe_serialization=True, 
        state_dict=sparse_state_dict
    )

print(f"Save compressed model peak GPU {m.overall_peak_memory / float(2**30):.4f} GB")

# use the dense state dict to reload the model
torch.cuda.set_device(1)
with measure_cuda_memory() as m:
    model_again = SparseAutoModelForCausalLM.from_pretrained(
        OUTPUT_PATH, 
        device_map="cuda:1"
    )

    #returns iterator
    dense_state_dict = compressor.decompress(OUTPUT_PATH)
    for name, data in tqdm(dense_state_dict, desc="Decompressing model"):
        BitmaskCompressor.replace_layer(name, data, model_again)

print(f"Load compressed model peak GPU {m.overall_peak_memory / float(2**30):.4f} GB")
```