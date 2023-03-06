# Token Classification: Sparse Transfer Learning with the CLI

In this example, you will fine-tune a 90% pruned BERT model onto some token classification datasets using SparseML's CLI.

### **Sparse Transfer Learning Overview**

Sparse Transfer Learning is very similiar to the typical transfer learing process used to train NLP models, where we fine-tune a pretrained checkpoint onto a smaller downstream dataset. With Sparse Transfer Learning, we simply start the fine-tuning process from a pre-sparsified checkpoint and maintain sparsity while the training process occurs.

### Pre-Sparsified BERT

SparseZoo, Neural Magic's open source repository of pre-sparsified models, contains a 90% pruned version of BERT, which has been sparsified on the upstream Wikipedia and BookCorpus datasets with the masked language modeling objective. [Check out the model card](https://sparsezoo.neuralmagic.com/models/nlp%2Fmasked_language_modeling%2Fobert-base%2Fpytorch%2Fhuggingface%2Fwikipedia_bookcorpus%2Fpruned90-none). We will use this model as the starting point for the transfer learning process.

**Let's dive in!**

## Installation

Install SparseML via `pip`.

```bash
pip install sparseml[torch]
```

## Sparse Transfer Learning onto Conll2003

SparseML's CLI enables you to kick-off sparsification workflows with various utilities like creating training pipelines, dataset loading, checkpoint saving, metric reporting, and logging handled for you. 

All we have to do is pass a couple of key arguments: 
- `--model_name_or_path` specifies the starting checkpoint to load for training
- `--task` specifies a glue dataset to train with 
- `--recipe` specifies path a recipe to use to apply sparsification algorithms or sparse transfer learning to the model. For Sparse Transfer Learning, we will use a recipe that instructs SparseML to maintain sparsity during the training process and to apply quantization over the final few epochs. 

### Run Transfer Learning

We will fine-tune a [90% pruned version of BERT](zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none) onto SST2.

Run the following:
```bash
sparseml.transformers.train.token_classification \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --recipe zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/pruned90_quant-none \
  --distill_teacher zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/base-none \
  --dataset_name conll2003 \
  --output_dir sparse_bert-token_classification_conll2003 \
  --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --preprocessing_num_workers 6 \
  --do_train --do_eval --evaluation_strategy epoch --fp16 --seed 29204  \
  --save_strategy epoch --save_total_limit 1
```

Let's discuss the key arguments:
- `--dataset_name conll2003` instructs SparseML to download and fine-tune onto the Conll2003 dataset. The script automatically downloads the dataset from the Hugging Face hub.

- `--model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none` specifies the starting checkpoint for the fine tuning. Here, we passed a SparseZoo stub identifying the 90% pruned version of BERT trained with masked language modeling, which SparseML downloads when the script starts.

- `--recipe zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/pruned90_quant-none` specifies the recipe to be applied by SparseML. Here, we passed a SparseZoo stub identifying the transfer learning recipe for the Conll2003 dataset, which SparseML downloads when the script starts. See below for the details of what this recipe looks like.

- `--distill_teacher zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/base-none` is an optional argument that specifies a model to use for as a teacher to apply distillation during the training process. We passed a SparseZoo stub identifying a dense BERT model trained on Conll2003, which SparseML downloads when the script starts.

The model trains for 13 epochs, converging to ~98.5% accuracy on the validation set. Because we applied a sparse transfer recipe, which instructs SparseML to maintain the sparsity of the starting pruned checkpoint and apply quantization, the final model is 90% pruned and quantized!

#### Transfer Learning Recipe

SparseML's recipes are YAML files that specify the sparsity related algorithms and parameters. SparseML parses the recipes and updates the training loops to apply the 
to apply sparsification algorithms or sparse transfer learning to the model.

In the case of Conll2003, we used a [premade recipe from the SparseZoo](https://sparsezoo.neuralmagic.com/models/nlp%2Ftoken_classification%2Fobert-base%2Fpytorch%2Fhuggingface%2Fconll2003%2Fpruned90_quant-none). 

<details>
    <summary>Click to inspect the recipe</summary>

The `Modifiers` are the important items that encode how SparseML should modify the training process for Sparse Transfer Learning:
- `ConstantPruningModifier` tells SparseML to pin weights at 0 over all epochs, maintaining the sparsity structure of the network
- `QuantizationModifier` tells SparseML to quanitze the weights with quantization aware training over the last 5 epochs
- `DistillationModifier` tells SparseML how to apply distillation during the trainign process, targeting the logits

SparseML parses the modifiers and updates the training process to implement the algorithms and hyperparameters specified in the recipes.

```yaml
version: 1.1.0

# General Variables
num_epochs: 13
init_lr: 1.5e-4 
final_lr: 0

qat_start_epoch: 8.0
observer_epoch: 12.0
quantize_embeddings: 1

distill_hardness: 1.0
distill_temperature: 2.0

# Modifiers:
training_modifiers:
  - !EpochRangeModifier
      end_epoch: eval(num_epochs)
      start_epoch: 0.0

  - !LearningRateFunctionModifier
      start_epoch: 0
      end_epoch: eval(num_epochs)
      lr_func: linear
      init_lr: eval(init_lr)
      final_lr: eval(final_lr)
    
quantization_modifiers:
  - !QuantizationModifier
      start_epoch: eval(qat_start_epoch)
      disable_quantization_observer_epoch: eval(observer_epoch)
      freeze_bn_stats_epoch: eval(observer_epoch)
      quantize_embeddings: eval(quantize_embeddings)
      quantize_linear_activations: 0
      exclude_module_types: ['LayerNorm']
      submodules:
        - bert.embeddings
        - bert.encoder
        - classifier

distillation_modifiers:
  - !DistillationModifier
     hardness: eval(distill_hardness)
     temperature: eval(distill_temperature)
     distill_output_keys: [logits]

constant_modifiers:
  - !ConstantPruningModifier
      start_epoch: 0.0
      params: __ALL_PRUNABLE__
```

</details>

You can download the recipe with the following code:

```python
from sparsezoo import Model
transfer_stub = "zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/pruned90_quant-none"
download_dir = "./transfer_recipe"
zoo_model = Model(transfer_stub, download_path=download_dir)
recipe_path = zoo_model.recipes.default.path
print(recipe_path)
```

### **Export to ONNX**

Once you have trained your model, export to ONNX in order to deploy with DeepSparse. The artifacts of the training process are saved to your local filesystem. 

Run the following to convert your PyTorch checkpoint to ONNX:

```bash
sparseml.transformers.export_onnx \
  --model_path ./sparse_bert-token_classification_conll2003 \
  --task text_classification
```

A `deployment` folder is created in your local directory, which has all of the files needed for deployment with DeepSparse including the `model.onnx`, `config.json`, and `tokenizer.json` files.

## Sparse Transfer Learning with a Custom Dataset (WNUT_17)

Beyond the Conll2003 dataset, we can also use a dataset from the Hugging Face Hub or pass via local files. Let's try an example of each for the sentiment analysis using [WNUT 17](wnut_17), which is also a NER task.

For simplicity, we will perform the fine-tuning without distillation. Although the transfer learning recipe contains distillation
modifiers, by setting `--distill_teacher disable` we instruct SparseML to skip distillation.

### WNUT_17 Inspection

Run the following to inspect the Rotten Tomatoes dataset.

```python
from datasets import load_dataset

wnut_17 = load_dataset("wnut_17")
print(wnut_17)
print(wnut_17["train"][0])
```

We can see that each row dataset contains a `tokens` field which contains a list of strings representing each word the sentence and a corresponding `ner_tags` which is a list of integers representing the tag of each word in the sentence.

### Using a Hugging Face Dataset Identifier

To use this dataset with the CLI, we can replace the `--dataset_name conll2003` argument with `--dataset_name wnut_17 --input_column_names tokens --label_column_name ner_tags`. SparseML will then download the dataset from the Hugging Face hub and run training as before.

```
sparseml.transformers.token_classification \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --recipe zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/pruned90_quant-none \
  --recipe_args '{"num_epochs":12,"qat_start_epoch":7.0, "observer_epoch": 11.0}' \
  --distill_teacher disable \
  --dataset_name wnut_17 --text_column_name tokens --label_column_name ner_tags \
  --output_dir sparse_bert-token_classification_wnut_17 \
  --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --preprocessing_num_workers 6 \
  --do_train --do_eval --evaluation_strategy epoch --fp16 --seed 29204  \
  --save_strategy epoch --save_total_limit 1
```

You will notice that we used the same recipe as we did in the Conll2003 case (identified by the SparseZoo stub `zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/pruned90_quant-none`). Since the WNUT and Conll2003 tasks are similiar, we chose to start with the same hyperparameters as we used in Conll2003 training.

#### Recipe Args

To update a recipe and experiment with hyperparameters, you can download the YAML file from SparseZoo, make updates to the YAML directly, and pass the local path to SparseML.

Alternatively, you can use `--recipe_args` to modify the recipe on the fly. In this case, we used the following to run for 12 epochs instead of 13:

```bash
--recipe_args '{"num_epochs":12,"qat_start_epoch":7.0, "observer_epoch": 11.0}'
```

### Using Local CSV/JSON Files

Let's walk through how to pass a CSV/JSON dataset to the CLI.

#### Save Dataset as a CSV File

For this example, we use Hugging Face `datasets` to create a JSON file for WNUT_17 that can be passed to SparseML's CLI but you can use any framework you want. For the Token Classification CLI, the label column must contain actual tags (i.e. not indexes). As such, we need to map the NER ids to tags before saving to JSON.

Run the following to create the CSV files:

```python
from datasets import load_dataset
from pprint import pprint

dataset = load_dataset("wnut_17")
print(dataset)
print(dataset["train"][0])

label_list = dataset["train"].features["ner_tags"].feature.names
print(label_list)

named_labels = []
for i in range(len(dataset["train"])):
  named_labels_i = [label_list[label_idx] for label_idx in dataset["train"][i]["ner_tags"]]
  named_labels.append(named_labels_i)

eval_named_labels = []
for i in range(len(dataset["validation"])):
  named_labels_i = [label_list[label_idx] for label_idx in dataset["validation"][i]["ner_tags"]]
  eval_named_labels.append(named_labels_i)

dataset["train"] = dataset["train"].add_column("named_ner_tags", named_labels)
dataset["validation"] = dataset["validation"].add_column("named_ner_tags", eval_named_labels)
dataset["train"] = dataset["train"].remove_columns("ner_tags")
dataset["validation"] = dataset["validation"].remove_columns("ner_tags")

dataset["train"].to_json("./wnut_17-train.json")
dataset["validation"].to_json("./wnut_17-validation.json")
```

We can see that the data is a JSON file with `tokens` and `named_ner_tags`. 

```bash
head ./wnut_17-train.json --lines=5
```

#### **Kick off Training**

To use the local files with the CLI, pass `--train_file ./wnut_17-train.json --validation_file ./wnut_17-validation.json  --text_column_name tokens --label_column_name named_ner_tags`.

Run the following:
```bash
sparseml.transformers.token_classification \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --recipe zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/pruned90_quant-none \
  --distill_teacher disable \
  --train_file wnut_17-train.json --validation_file wnut_17-validation.json \
  --text_column_name tokens --label_column_name named_ner_tags \
  --output_dir sparse_bert-token_classification_wnut_17_from_json \
  --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --preprocessing_num_workers 6 \
  --do_train --do_eval --evaluation_strategy epoch --fp16 --seed 29204  \
  --save_strategy epoch --save_total_limit 1
```

### Sparse Transfer Learning with a Custom Teacher

Stay tuned for an example with a custom teacher.