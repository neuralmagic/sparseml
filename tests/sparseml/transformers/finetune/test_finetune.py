from sparseml.transformers.finetune.text_generation import run_general
import torch

def test_oneshot_and_finetune(tmp_path):
    recipe_str = "tests/sparseml/transformers/finetune/test_alternate_recipe.yaml"
    model = "Xenova/llama2.c-stories15M"
    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"
    dataset_name = "wikitext"
    dataset_config_name = "wikitext-2-raw-v1"
    concatenate_data = True
    run_stages=True
    output_dir = tmp_path
    max_steps = 50
    splits = {
        "train": "train[:50%]",
        "calibration": "train[50%:60%]"
    }

    run_general(
        model_name_or_path=model,
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        run_stages=run_stages,
        output_dir=output_dir,
        recipe=recipe_str,
        max_steps=max_steps,
        concatenate_data = concatenate_data,
        splits = splits,
        oneshot_device = device
    )


