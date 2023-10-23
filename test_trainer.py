from sparseml.transformers.text_generation import main

model = "facebook/opt-350m"
dataset_name = "wikitext"
dataset_config_name = "wikitext-2-raw-v1"
do_train = True
do_eval = False
output_dir = "./output"
cache_dir = "cache"
recipe = "test_trainer_recipe.yaml"
recipe_args = {
    "num_epochs": 7,
    "pruning_init_sparsity": 0.0,
    "pruning_final_sparsity": 0.5,
    "pruning_start_epoch": 2,
    "pruning_end_epoch": 5
}
num_train_epochs=7

main(
    model_name_or_path=model,
    dataset_name=dataset_name,
    dataset_config_name=dataset_config_name,
    do_train=do_train,
    do_eval=do_eval,
    output_dir=output_dir,
    cache_dir=cache_dir,
    recipe=recipe,
    recipe_args=recipe_args,
    max_train_samples = 1024,
    max_eval_samples = 256,
    num_train_epochs=num_train_epochs
)