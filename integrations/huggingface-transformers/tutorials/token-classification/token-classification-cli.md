<!--
Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Token Classification: Sparse Transfer Learning with the CLI

In this example, you will fine-tune a 90% pruned BERT model onto some token classification datasets using SparseML's CLI.

### **Sparse Transfer Learning Overview**

Sparse Transfer Learning is very similiar to the typical transfer learing process used to train NLP models, where we fine-tune a pretrained checkpoint onto a smaller downstream dataset. With Sparse Transfer Learning, we simply start the fine-tuning process from a pre-sparsified checkpoint and maintain sparsity while the training process occurs.

### Pre-Sparsified BERT

SparseZoo, Neural Magic's open source repository of pre-sparsified models, contains a 90% pruned version of BERT, which has been sparsified on the upstream Wikipedia and BookCorpus datasets with the masked language modeling objective.  We will use this model as the starting point for the transfer learning process.

- [Check out 90% pruned BERT model card](https://sparsezoo.neuralmagic.com/models/nlp%2Fmasked_language_modeling%2Fobert-base%2Fpytorch%2Fhuggingface%2Fwikipedia_bookcorpus%2Fpruned90-none)
- [Check out the full list of pre-sparsified NLP models](https://sparsezoo.neuralmagic.com/?domain=nlp&sub_domain=masked_language_modeling&page=1)

### Table of Contents

In this tutorial, you will learn how to:
- [Sparse Transfer Learn onto Conll2003](#sparse-transfer-learning-onto-conll2003)
- [Sparse Transfer Learn onto a Custom Dataset (WNut17)](#sparse-transfer-learning-with-a-custom-dataset-wnut_17)
- [Sparse Transfer Learn with a Custom Teacher](#sparse-transfer-learning-with-a-custom-teacher)

## Installation

Install SparseML via `pip`:

```bash
pip install sparseml[transformers]
```

## Sparse Transfer Learning onto Conll2003

SparseML's CLI offers pre-made training pipelines for common NLP tasks, including token classification.

The CLI enables you to kick-off training runs with various utilities like dataset loading and pre-processing, checkpoint saving, metric reporting, and logging handled for you.

All we have to do is pass a couple of key arguments: 
- `--model_name_or_path` specifies the starting checkpoint to load for training
- `--dataset_name` specifies a Hugging Face dataset to train with 
- `--recipe` specifies path a recipe to use to apply sparsification algorithms or sparse transfer learning to the model. For Sparse Transfer Learning, we will use a recipe that instructs SparseML to maintain sparsity during the training process and to apply quantization over the final few epochs. 

### Create a Transfer Learning Recipe

To launch a Sparse Transfer Learning run, we first need to create a Sparse Transfer Learning recipe.

Recipes are YAML files that specify sparsity related algorithms and hyper-parameters. SparseML parses the recipes and updates the training loops to apply the specified sparsification algorithms to the model.

In the case of Conll2003, there is a [premade recipe from the SparseZoo](https://sparsezoo.neuralmagic.com/models/nlp%2Ftoken_classification%2Fobert-base%2Fpytorch%2Fhuggingface%2Fconll2003%2Fpruned90_quant-none):

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

The `Modifiers` are the important items that encode how SparseML should modify the training process for Sparse Transfer Learning:
- `ConstantPruningModifier` tells SparseML to pin weights at 0 over all epochs, maintaining the sparsity structure of the network
- `QuantizationModifier` tells SparseML to quanitze the weights with quantization aware training over the last 5 epochs
- `DistillationModifier` tells SparseML how to apply distillation during the training process, targeting the logits

SparseML parses the modifiers and updates the training process to implement the algorithms and hyperparameters specified in the recipes.

You can download the recipe with the following code:

```python
from sparsezoo import Model
transfer_stub = "zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/pruned90_quant-none"
download_dir = "./transfer_recipe"
zoo_model = Model(transfer_stub, download_path=download_dir)
recipe_path = zoo_model.recipes.default.path
print(recipe_path)
```

### Fine Tune the Model

With the recipe and starting sparse checkpoint identified, we can kick off the fine-tuning with the following:

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

### **Export to ONNX**

Once you have trained your model, export to ONNX in order to deploy with DeepSparse with th following:

```bash
sparseml.transformers.export_onnx \
  --model_path ./sparse_bert-token_classification_conll2003 \
  --task token_classification
```

A `deployment` folder is created in your local directory, which has all of the files needed for deployment with DeepSparse including the `model.onnx`, `config.json`, and `tokenizer.json` files.

## Sparse Transfer Learning with a Custom Dataset (WNUT_17)

Beyond the Conll2003 dataset, we can also use a dataset from the Hugging Face Hub or from local files. Let's try an example of each for the sentiment analysis using WNUT_17, which is also a NER task.

For simplicity, we will perform the fine-tuning without distillation. Although the transfer learning recipe contains distillation
modifiers, by setting `--distill_teacher disable` we instruct SparseML to skip distillation.

### WNUT_17 Inspection

Run the following to inspect the Rotten Tomatoes dataset.

```python
from datasets import load_dataset

wnut_17 = load_dataset("wnut_17")
print(wnut_17)
print(wnut_17["train"][0])

# > {'id': '0', 'tokens': ['@paulwalk', 'It', "'s", 'the', 'view', 'from', 'where', 'I', "'m", 'living', 'for', 'two', 'weeks', '.', 'Empire', 'State', 'Building', '=', 'ESB', '.', 'Pretty', 'bad', 'storm', 'here', 'last', 'evening', '.'], 'ner_tags': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 8, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0]}
```

We can see that each row contains a `tokens` field which contains a list of strings representing each word the sentence and a corresponding `ner_tags` which is a list of integers representing the tag of each word in the sentence.

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

### Using Local JSON Files

Let's walk through how to pass a JSON dataset to the CLI.

#### Save Dataset as a JSON File

We use Hugging Face `datasets` to create a JSON file for WNUT_17 that can be passed to SparseML's CLI. 

For the Token Classification CLI, the label column must contain actual tags (i.e. not indexes). As such, we need to map the NER ids to tags before saving to JSON.

Run the following to create the JSON files:

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

Output:

```bash
{"id":"0","tokens":["@paulwalk","It","'s","the","view","from","where","I","'m","living","for","two","weeks",".","Empire","State","Building","=","ESB",".","Pretty","bad","storm","here","last","evening","."],"named_ner_tags":["O","O","O","O","O","O","O","O","O","O","O","O","O","O","B-location","I-location","I-location","O","B-location","O","O","O","O","O","O","O","O"]}
{"id":"1","tokens":["From","Green","Newsfeed",":","AHFA","extends","deadline","for","Sage","Award","to","Nov",".","5","http:\/\/tinyurl.com\/24agj38"],"named_ner_tags":["O","O","O","O","B-group","O","O","O","O","O","O","O","O","O","O"]}
{"id":"2","tokens":["Pxleyes","Top","50","Photography","Contest","Pictures","of","August","2010","...","http:\/\/bit.ly\/bgCyZ0","#photography"],"named_ner_tags":["B-corporation","O","O","O","O","O","O","O","O","O","O","O"]}
{"id":"3","tokens":["today","is","my","last","day","at","the","office","."],"named_ner_tags":["O","O","O","O","O","O","O","O","O"]}
{"id":"4","tokens":["4Dbling","'s","place","til","monday",",","party","party","party",".","&lt;","3"],"named_ner_tags":["B-person","O","O","O","O","O","O","O","O","O","O","O"]}
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

## Sparse Transfer Learning with a Custom Teacher

To increase accuracy, we can apply model distillation from a dense teacher model, just like we did for the Conll2003 case. You are free to use the native Hugging Face workflows to train the dense teacher model (and can even pass a Hugging Face model identifier to the --distill_teacher argument), but can also use the SparseML CLI.

### Train the Dense Teacher

Run the follwing to train a dense model on WNUT (using the data files from above):

```bash
sparseml.transformers.token_classification \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/base-none \
  --recipe zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/base-none \
  --distill_teacher disable \
  --train_file wnut_17-train.json --validation_file wnut_17-validation.json \
  --text_column_name tokens --label_column_name named_ner_tags \
  --output_dir wnut_dense_teacher \
  --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --preprocessing_num_workers 6 \
  --do_train --do_eval --evaluation_strategy epoch --fp16 --seed 29204  \
  --save_strategy epoch --save_total_limit 1
```

Note that used the dense version of BERT (the stub ends in base-none) as the starting point for the training and passed a recipe from SparseZoo which was used to train the dense teacher for the Conll2003 task. Since the Conll2003 task is similiar to the WNUT task, these parameters are a solid place to start. This recipe contains no sparsity related modifiers and only controls the learning rate and number of epochs. As such, the script will run typical fine-tuning, resulting in a dense model.

Here is what the recipe looks like:

```yaml
version: 1.1.0

# General Variables
num_epochs: 4
init_lr: 5e-5
final_lr: 0

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
```

### Fine Tune with a Custom Teacher

With the dense teacher trained, we can sparse transfer learn with the help of the teacher by passing a local path to the model checkpoint. In this case, we use `--distill_teacher ./wnut_dense_teacher`.

Run the following to kick off training:

```bash
sparseml.transformers.token_classification \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --recipe zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/pruned90_quant-none \
  --recipe_args '{"num_epochs": 10, "qat_start_epoch": 5.0, "observer_epoch": 9.0}' \
  --distill_teacher ./wnut_dense_teacher \
  --train_file wnut_17-train.json --validation_file wnut_17-validation.json \
  --text_column_name tokens --label_column_name named_ner_tags \
  --output_dir wnut_sparse_with_teacher \
  --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --preprocessing_num_workers 6 \
  --do_train --do_eval --evaluation_strategy epoch --fp16 --seed 29204  \
  --save_strategy epoch --save_total_limit 1
```

The resulting model is 90% pruned and quantized.
