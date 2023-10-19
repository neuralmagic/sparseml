from sparseml.transformers.text_generation import main

model = "facebook/opt-350m"
dataset_name = "wikitext2"
nsamples = 1024
recipe = "test_qa_recipe.yaml"

main(
    model_path=model,
    dataset_name=dataset_name,
    num_samples=nsamples,
    recipe_file=recipe
)