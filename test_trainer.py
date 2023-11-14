def run():
    from sparseml.transformers.finetune.text_generation import main
    
    model = "./obcq_deployment"
    dataset_name = "c4"
    dataset_config_name = "allenai--c4"
    concatenate_data = None
    do_train = True
    do_eval = False
    output_dir = "./output_finetune"
    recipe = "test_trainer_recipe.yaml"
    num_train_epochs=2
    overwrite_output_dir = True
    raw_kwargs = {"data_files": {"train": "en/c4-train.00000-of-01024.json.gz"}}
    splits = {
        "train": "train[:5%]",
    }

    main(
        model_name_or_path=model,
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        do_train=do_train,
        do_eval=do_eval,
        output_dir=output_dir,
        recipe=recipe,
        num_train_epochs=num_train_epochs,
        overwrite_output_dir=overwrite_output_dir,
        concatenate_data = concatenate_data,
        splits = splits,
        raw_kwargs=raw_kwargs
    )

if __name__ == "__main__":
    run()