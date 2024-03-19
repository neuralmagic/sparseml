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
from sparseml.transformers import SparseAutoModelForCausalLM, SparseAutoTokenizer
from sparseml.utils.pytorch.utils import measure_cuda_memory
import torch

MODEL_PATH = "zoo:llama2-7b-open_platypus_orca_llama2_pretrain-pruned60"
OUTPUT_PATH = "./test_compress_output"
RECIPE = "zoo:llama2-7b-open_platypus_orca_llama2_pretrain-pruned60"

torch.cuda.set_device(0)
with measure_cuda_memory() as m:
    model = SparseAutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="cuda:0")
print(f"Load dense model peak GPU {m.overall_peak_memory / float(2**30):.4f} GB")
tokenizer = SparseAutoTokenizer.from_pretrained(MODEL_PATH)

print(f"Sparsity config before compression: {model.sparsity_config}")
with measure_cuda_memory() as m:
    SparseAutoModelForCausalLM.save_pretrained(
        model, OUTPUT_PATH
    )
print(f"Save compressed model peak GPU {m.overall_peak_memory / float(2**30):.4f} GB")

torch.cuda.set_device(1)
with measure_cuda_memory() as m:
    model_again = SparseAutoModelForCausalLM.from_pretrained(
        OUTPUT_PATH, device_map="cuda:1"
    )
print(f"Load compressed model peak GPU {m.overall_peak_memory / float(2**30):.4f} GB")
print(f"Sparsity config after compression: {model_again.sparsity_config}")
```