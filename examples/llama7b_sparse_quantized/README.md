# Creating a Sparse Quantized Llama7b Model

The example in this folder runs in multiple stages to create a Llama 7b model with 
a 2:4 sparsity pattern and W4A16 quantization. The model is calibrated and trained with 
the Ultra Chat 200k dataset.

## Recipe Summary

The recipe used for this flow is located in [2:4_w4a16_recipe.yaml](/2:4_w4a16_recipe.yaml). It contains 3 stages that are outlined below.


### Stage 1: Sparsification

Runs the SparseGPT one-shot algorithm to prune the model to 50% sparsity with a 2:4 
sparsity pattern. This means that 2 weights out of every group of 4 weights are masked to 0.

### Stage 2: Finetuning Recovery

This stage runs a single epoch of training on the ultrachat200k dataset while maintaining 
the sparsity mask from stage 1. The purpose of this stage is to recover any accuracy lost 
during the sparsification process.

### Stage 3: Quantization

Finally, we run the GPTQ one-shot algorithm to quantize all linear weights to 4 bit 
channelwise.

## How to Run

We can run the entire staged recipe with one call to SparseML's `compress` pathway. This 
will save a checkpoint of the model after each stage.

```python examples/llama7b_sparse_quantized/llama7b_sparse_w4a16.py```

### Running with FSDP (Experimental)

This pathway can also be run in sharded mode across multiple GPUs using Pytorch FSDP. To
launch the FSDP version of the example script, update the `num_processes` field in
[fsdp_config.yaml](fsdp_config.yaml) with the number of GPUs on your machine. Then run:

```bash run_recipe_fsdp.sh```

For more information on setting up FSDP configs, see the [FSDP Configuration README](../integrations/huggingface-transformers/finetuning/configure_fsdp.md)

### Compression

The resulting model will be uncompressed. To save a final compressed copy of the model 
run the following:

```
import torch
from sparseml import SparseAutoModelForCausalLM

model = SparseAutoModelForCausalLM.from_pretrained(output_dir, torch_dtype=torch.bfloat16)
model.save_pretrained(compressed_output_dir, save_compressed=True)
```
