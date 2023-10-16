from sparseml.transformers.question_answering import main

model = "bert-base-uncased"
dataset_name = "squad"
do_train = True
do_eval = True
output_dir = "./output"
cache_dir = "cache"
distill_teacher = "disable"
recipe = "test_qa_recipe.yaml"
recipe_args = {
    "num_epochs": 30,
    "pruning_init_sparsity": 0.7,
    "pruning_final_sparsity": 0.9,
    "pruning_start_epoch": 2,
    "pruning_end_epoch": 26
}

main(
    model_name_or_path=model,
    dataset_name=dataset_name,
    do_train=do_train,
    do_eval=do_eval,
    output_dir=output_dir,
    cache_dir=cache_dir,
    distill_teacher=distill_teacher,
    recipe=recipe,
    recipe_args=recipe_args
)