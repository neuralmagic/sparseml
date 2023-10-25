from sparseml.transformers.finetune.text_generation import main

def run():
    model = "./obcq_deployment_tiny_llama"
    dataset_name = "wikitext"
    dataset_config_name = "wikitext-2-raw-v1"
    concatenate_data = True
    do_train = True
    do_eval = False
    output_dir = "./output"
    cache_dir = "cache"
    recipe = "test_trainer_recipe.yaml"
    num_train_epochs=1
    overwrite_output_dir = True

    main(
        model_name_or_path=model,
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        do_train=do_train,
        do_eval=do_eval,
        output_dir=output_dir,
        cache_dir=cache_dir,
        recipe=recipe,
        max_train_samples = 32,
        max_eval_samples = 16,
        num_train_epochs=num_train_epochs,
        overwrite_output_dir=overwrite_output_dir,
        concatenate_data = concatenate_data
    )

if __name__ == "__main__":
    run()