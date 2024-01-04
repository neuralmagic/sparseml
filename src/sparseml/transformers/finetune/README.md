# Sparse Finetuning

## Launching from Console Scripts

### with DataParallel (default)

```bash
sparseml.transformers.text_generation.train
    --model_name PATH_TO_MODEL
    --distill_teacher PATH_TO_TEACHER
    --dataset_name DATASET_NAME
    --recipe PATH_TO_RECIPE
    --output_dir PATH_TO_OUTPUT
    --num_train_epochs 1
    --splits "train"
```

Also supported:

* `sparseml.transformers.text_generation.finetune`
* `sparseml.transformers.text_generation.eval`

### with FSDP

```bash
accelerate launch 
    --config_file example_fsdp_config.yaml 
    --no_python sparseml.transformers.text_generation.finetune
    --model_name PATH_TO_MODEL
    --distill_teacher PATH_TO_TEACHER
    --dataset_name DATASET_NAME
    --recipe PATH_TO_RECIPE
    --output_dir PATH_TO_OUTPUT
    --num_train_epochs 1
    --splits "train"
```

See [configure_fsdp.md](https://github.com/neuralmagic/sparseml/blob/main/integrations/huggingface-transformers/finetuning/configure_fsdp.md) for additional instructions on setting up FSDP configuration

## Launching from Python

```python
from sparseml.transformers.finetune.text_generation import run_train

model = "./obcq_deployment"
teacher_model = "Xenova/llama2.c-stories15M"
dataset_name = "open_platypus"
concatenate_data = False
output_dir = "./output_finetune"
recipe = "test_trainer_recipe.yaml"
num_train_epochs=2
overwrite_output_dir = True
splits = {
    "train": "train[:50%]",
}

run_train(
    model_name_or_path=model,
    distill_teacher=teacher_model,
    dataset_name=dataset_name,
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
