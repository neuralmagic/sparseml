from datasets import load_dataset


# path = "scratch/data1.json"
# path = "scratch/data.json"
# path = "scratch/train/data1.json"
# pp = load_dataset('json', data_files=path)
# print(pp)

path = "tests/sparseml/transformers/finetune/data/train/"
data_files ={
    "train": [
        path + "data1.json",
        path + 'data2.json',
    ],
    "test": [
        path + "data1.json",
        path + 'data2.json',
    ]
}

pp = load_dataset('json', data_files=data_files)
print(pp)
