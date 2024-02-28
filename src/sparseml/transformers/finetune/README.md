# Sparse Finetuning

## Launching from Console Scripts

### with DataParallel (default)

```bash
sparseml.transformers.text_generation.train
    --model PATH_TO_MODEL
    --distill_teacher PATH_TO_TEACHER
    --dataset DATASET
    --recipe PATH_TO_RECIPE
    --output_dir PATH_TO_OUTPUT
    --num_train_epochs 1
    --splits "train"
```

Also supported:

* `sparseml.transformers.text_generation.finetune`
* `sparseml.transformers.text_generation.oneshot`
* `sparseml.transformers.text_generation.eval`
* `sparseml.transformers.text_generation.apply`

### with FSDP

```bash
accelerate launch 
    --config_file example_fsdp_config.yaml 
    --no_python sparseml.transformers.text_generation.finetune
    --model PATH_TO_MODEL
    --distill_teacher PATH_TO_TEACHER
    --dataset DATASET
    --recipe PATH_TO_RECIPE
    --output_dir PATH_TO_OUTPUT
    --num_train_epochs 1
    --splits "train"
```

See [configure_fsdp.md](https://github.com/neuralmagic/sparseml/blob/main/integrations/huggingface-transformers/finetuning/configure_fsdp.md) for additional instructions on setting up FSDP configuration

## Launching from Python

```python
from sparseml.transformers import train

model = "./obcq_deployment"
teacher_model = "Xenova/llama2.c-stories15M"
dataset = "open_platypus"
concatenate_data = False
output_dir = "./output_finetune"
recipe = "test_trainer_recipe.yaml"
num_train_epochs=2
overwrite_output_dir = True
splits = {
    "train": "train[:50%]",
}

train(
    model=model,
    distill_teacher=teacher_model,
    dataset=dataset,
    output_dir=output_dir,
    recipe=recipe,
    num_train_epochs=num_train_epochs,
    overwrite_output_dir=overwrite_output_dir,
    concatenate_data = concatenate_data,
    splits = splits
)
```

## Additional Configuration

Finetuning arguments are split up into 3 groups:

* ModelArguments: `src/sparseml/transformers/finetune/model_args.py`
* TrainingArguments: `src/sparseml/transformers/finetune/training_args.py`
* DataTrainingArguments: `src/sparseml/transformers/finetune/data/data_training_args.py`


## Running One-Shot with FSDP
```bash
accelerate launch 
    --config_file example_fsdp_config.yaml 
    --no_python sparseml.transformers.text_generation.oneshot
    --model PATH_TO_MODEL
    --num_calibration_samples 512
    --dataset DATASET
    --dataset_config_name OPTIONAL
    --max_seq_len OPTIONAL
    --concatenate_data OPTIONAL
    --recipe PATH_TO_RECIPE
    --output_dir PATH_TO_OUTPUT
    --splits "train"
```


## Running One-shot from Python (without FSDP)
```python
from sparseml.transformers import oneshot

model = "Xenova/llama2.c-stories15M"
dataset = "open_platypus"
concatenate_data = False
output_dir = "./output_oneshot"
recipe = "test_oneshot_recipe.yaml"
overwrite_output_dir = True
splits = {
    "calibration": "train[:20%]"
}

oneshot(
    mode=model,
    dataset=dataset,
    concatenate_data=concatenate_data,
    output_dir=output_dir,
    recipe=recipe,
    overwrite_output_dir=overwrite_output_dir,
    concatenate_data = concatenate_data,
    splits = splits
)
```

## Running Multi-Stage Recipes

A recipe can be run stage-by-stage by setting `run_stages` to `True`. Each stage in the
recipe should have a `run_type` attribute set to either `oneshot` or `train`.

See [example_alternating_recipe.yaml](example_alternating_recipe.yaml) for an example 
of a staged recipe for Llama. 

### Python Example
(This can also be run with FSDP by launching the script as `accelerate launch --config_file example_fsdp_config.yaml test_multi.py`)

test_multi.py
```python
from sparseml.transformers import apply

model = "../ml-experiments/nlg-text_generation/llama_pretrain-llama_7b-base/dense/training"
dataset = "open_platypus"
concatenate_data = False
run_stages=True
output_dir = "./output_finetune_multi"
recipe = "example_alternating_recipe.yaml"
num_train_epochs=1
overwrite_output_dir = True
splits = {
    "train": "train[:95%]",
    "calibration": "train[95%:100%]"
}

apply(
    model_or_path=model,
    dataset=dataset,
    run_stages=run_stages,
    output_dir=output_dir,
    recipe=recipe,
    num_train_epochs=num_train_epochs,
    overwrite_output_dir=overwrite_output_dir,
    concatenate_data = concatenate_data,
    remove_unused_columns = False,
    splits = splits
)
```