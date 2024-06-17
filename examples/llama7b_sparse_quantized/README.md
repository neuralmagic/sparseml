# Creating a Sparse Quantized Llama7b Model

This example uses SparseML and Compressed-Tensors to create a 2:4 sparse and quantized Llama2-7b model.
The model is calibrated and trained with the ultachat200k dataset.
At least 75GB of GPU memory is required to run this example.

Follow the steps below one by one in a code notebook, or run the full example script 
as `python examples/llama7b_sparse_quantized/llama7b_sparse_w4a16.py`

## Step 1: Select a model, dataset, and recipe
In this step, we select which model to use as a baseline for sparsification, a dataset to
use for calibration and finetuning, and a recipe.

Models can reference a local directory, model in the huggingface hub, or in the sparsezoo.

Datasets can be from a local compatible directory or the huggingface hub.

Recipes are YAML files that describe how a model should be optimized during or after training.
The recipe used for this flow is located in [2:4_w4a16_recipe.yaml](./2:4_w4a16_recipe.yaml).
It contains instructions to prune the model to 2:4 sparsity, run one epoch of recovery finetuning,
and quantize to 4 bits in one show using GPTQ.

```python
import torch
from sparseml.transformers import SparseAutoModelForCausalLM

model_stub = "zoo:llama2-7b-ultrachat200k_llama2_pretrain-base"
model = SparseAutoModelForCausalLM.from_pretrained(
    model_stub, torch_dtype=torch.bfloat16, device_map="auto"
)

dataset = "ultrachat-200k"
splits = {"calibration": "train_gen[:5%]", "train": "train_gen"}

recipe = "2:4_w4a16_recipe.yaml"
```

## Step 2: Run sparsification using `apply`
The `apply` function applies the given recipe to our model and dataset.
The hardcoded kwargs may be altered based on each model's needs. This code snippet should 
be run in the same Python instance as step 1.
After running, the sparsified model will be saved to `output_llama7b_2:4_w4a16_channel`.

```python
from sparseml.transformers import apply

output_dir = "output_llama7b_2:4_w4a16_channel"

apply(
    model=model,
    dataset=dataset,
    recipe=recipe,
    bf16=False,  # use full precision for training
    output_dir=output_dir,
    splits=splits,
    max_seq_length=512,
    num_calibration_samples=512,
    num_train_epochs=0.5,
    logging_steps=500,
    save_steps=5000,
    gradient_checkpointing=True,
    learning_rate=0.0001,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
)
```


### Step 3: Compression

The resulting model will be uncompressed. To save a final compressed copy of the model 
run the following in the same Python instance as the previous steps.

```python
import torch
from sparseml.transformers import SparseAutoModelForCausalLM

compressed_output_dir = "output_llama7b_2:4_w4a16_channel_compressed"
model = SparseAutoModelForCausalLM.from_pretrained(output_dir, torch_dtype=torch.bfloat16)
model.save_pretrained(compressed_output_dir, save_compressed=True)
```

### Custom Quantization
The current repo supports multiple quantization techniques configured using a recipe. Supported strategies are `tensor`, `group` and `channel`. 
The above recipe (`2:4_w4a16_recipe.yaml`) uses channel-wise quantization specified by `strategy: "channel"` in its config group. 
To use quantize per tensor, change strategy from `channel` to `tensor`. To use group size quantization, change from `channel` to `group` and specify its value, say 128, by including `group_size: 128`. A group size quantization example is shown in `2:4_w4a16_group-128_recipe.yaml`.
