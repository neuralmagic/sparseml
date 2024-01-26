from sparseml.transformers.finetune.data.data_helpers import get_custom_datasets_from_path


path = 'scratch/dataset'
xx = get_custom_datasets_from_path(path)
print(xx)
print()
print()
print()
print()
path = 'scratch/dataset2'
xx = get_custom_datasets_from_path(path)
print(xx)

