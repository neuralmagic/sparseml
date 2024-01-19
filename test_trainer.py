def run():
    from sparseml.transformers.finetune.text_generation import run_general
    
    model = "../ml-experiments/nlg-text_generation/llama_pretrain-llama_7b-base/dense/training"
    dataset_name = "open_platypus"
    concatenate_data = False
    do_train = True
    do_eval = False
    output_dir = "./output_finetune"
    recipe = None #"test_trainer_recipe.yaml"
    num_train_epochs=2
    overwrite_output_dir = True
    splits = {
        "train": "train[:90%]",
        "validation": "train[90%:]"
    }

    run_general(
        model_name_or_path=model,
        dataset_name=dataset_name,
        do_train=do_train,
        do_eval=do_eval,
        output_dir=output_dir,
        recipe=recipe,
        num_train_epochs=num_train_epochs,
        overwrite_output_dir=overwrite_output_dir,
        concatenate_data = concatenate_data,
        remove_unused_columns = False,
        splits = splits
    )

if __name__ == "__main__":
    run()