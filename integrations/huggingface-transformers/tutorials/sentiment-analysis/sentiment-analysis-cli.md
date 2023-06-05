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

# Sentiment Analysis: Sparse Transfer Learning with the CLI

In this example, you will fine-tune a 90% pruned BERT model onto some sentiment-analysis datasets using SparseML's CLI.

### **Sparse Transfer Learning Overview**

Sparse Transfer Learning is very similiar to the typical transfer learning process used to train NLP models, where we fine-tune a pretrained checkpoint onto a smaller downstream dataset. With Sparse Transfer Learning, however, we simply start the fine-tuning process from a pre-sparsified checkpoint and maintain sparsity while the training process occurs.

### Pre-Sparsified BERT

SparseZoo, Neural Magic's open source repository of pre-sparsified models, contains a 90% pruned version of BERT, which has been sparsified on the upstream Wikipedia and BookCorpus datasets with the masked language modeling objective.  We will use this model as the starting point for the transfer learning process.

- [Check out 90% pruned BERT model card](https://sparsezoo.neuralmagic.com/models/nlp%2Fmasked_language_modeling%2Fobert-base%2Fpytorch%2Fhuggingface%2Fwikipedia_bookcorpus%2Fpruned90-none)
- [Check out the full list of pre-sparsified NLP models](https://sparsezoo.neuralmagic.com/?domain=nlp&sub_domain=masked_language_modeling&page=1)

### Table of Contents

In this tutorial, you will learn how to:
- [Sparse Transfer Learn onto a GLUE task (SST2)](#sparse-transfer-learning-onto-sst2-glue-task)
- [Sparse Transfer Learn onto a Custom Dataset (Rotten Tomatoes)](#sparse-transfer-learning-with-a-custom-dataset-rotten-tomatoes)
- [Sparse Transfer Learn with a Custom Teacher (Rotten Tomatoes)](#sparse-transfer-learning-with-a-custom-teacher-rotten-tomatoes)

## Installation

Install SparseML via `pip`:

```bash
pip install sparseml[transformers]
```

## Sparse Transfer Learning onto SST2 (GLUE Task)

SparseML's CLI offers pre-made training pipelines for common NLP tasks, including text classification. 

The CLI enables you to kick-off training runs with various utilities like dataset loading and pre-processing, checkpoint saving, metric reporting, and logging handled for you.

All we have to do is pass a couple of key arguments: 
- `--model_name_or_path` specifies the starting checkpoint to load for training
- `--task` specifies a glue task to train on
- `--recipe` specifies path a recipe to use to apply sparsification algorithms or sparse transfer learning to the model. For Sparse Transfer Learning, we will use a recipe that instructs SparseML to maintain sparsity during the training process and to apply quantization over the final few epochs. 

### Create a Transfer Learning Recipe

To launch a Sparse Transfer Learning run, we first need to create a Sparse Transfer Learning recipe.

Recipes are YAML files that specify sparsity related algorithms and hyper-parameters. SparseML parses the recipes and updates the training loops to apply the specified sparsification algorithms to the model.

In the case of SST2, there is a [premade recipe in the SparseZoo](https://sparsezoo.neuralmagic.com/models/nlp%2Fsentiment_analysis%2Fobert-base%2Fpytorch%2Fhuggingface%2Fsst2%2Fpruned90_quant-none):

```yaml
version: 1.1.0

# General Variables
num_epochs: &num_epochs 13
init_lr: 1.5e-4
final_lr: 0

qat_start_epoch: &qat_start_epoch 8.0
observer_epoch: &observer_epoch 12.0
quantize_embeddings: &quantize_embeddings 1

distill_hardness: &distill_hardness 1.0
distill_temperature: &distill_temperature 2.0

weight_decay: 0.01

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
      exclude_module_types: ['LayerNorm', 'Tanh']
      submodules:
        - bert.embeddings
        - bert.encoder
        - bert.pooler
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

regularization_modifiers:
  - !SetWeightDecayModifier
      start_epoch: 0.0
      weight_decay: eval(weight_decay)
```

The `Modifiers` are the important items that encode how SparseML should modify the training process for Sparse Transfer Learning:
- `ConstantPruningModifier` tells SparseML to pin weights at 0 over all epochs, maintaining the sparsity structure of the network
- `QuantizationModifier` tells SparseML to quanitze the weights with quantization aware training over the last 5 epochs
- `DistillationModifier` tells SparseML how to apply distillation during the training process, targeting the logits

SparseML parses the modifiers and updates the training process to implement the algorithms and hyperparameters specified in the recipes.

You can download the recipe with the following code:

```python
from sparsezoo import Model
transfer_stub = "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none"
download_dir = "./transfer_recipe"
zoo_model = Model(transfer_stub, download_path=download_dir)
recipe_path = zoo_model.recipes.default.path
print(recipe_path)
```

### Fine Tune The Model

With the recipe and starting sparse checkpoint identified, we can kick off the fine-tuning with the following:
```bash
sparseml.transformers.text_classification \
  --task_name sst2 \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --recipe zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none \
  --distill_teacher zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/base-none \
  --output_dir sparse_quantized_bert-text_classification_sst2 \
  --max_seq_length 128 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --preprocessing_num_workers 6 \
  --do_train --do_eval --evaluation_strategy epoch --fp16 --save_strategy epoch --save_total_limit 1
```

Let's discuss the key arguments:
- `--task_name sst2` instructs SparseML to download and fine-tune onto the SST2 dataset. You can pass any GLUE task to this parameter.

- `--model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none` specifies the starting checkpoint for the fine tuning. Here, we passed a SparseZoo stub identifying the 90% pruned version of BERT trained with masked language modeling, which SparseML downloads when the script starts.

- `--recipe zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none` specifies the recipe to be applied by SparseML. Here, we passed a SparseZoo stub identifying the transfer learning recipe for the SST2 dataset, which SparseML downloads when the script starts.

- `--distill_teacher zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/base-none` is an optional argument that specifies a model to use for as a teacher to apply distillation during the training process. Here, we passed a SparseZoo stub identifying a dense BERT model trained on SST2, which SparseML downloads when the script starts.

The model trains for 13 epochs, converging to ~92% accuracy on the validation set. Because we applied a sparse transfer recipe, which instructs SparseML to maintain the sparsity of the starting pruned checkpoint and apply quantization, the final model is 90% pruned and quantized!

### **Export to ONNX**

Once you have trained your model, export to ONNX in order to deploy with DeepSparse with the following:

```bash
sparseml.transformers.export_onnx \
  --model_path sparse_quantized_bert-text_classification_sst2 \
  --task text_classification
```

A `deployment` folder is created in your local directory, which has all of the files needed for deployment with DeepSparse including the `model.onnx`, `config.json`, and `tokenizer.json` files.

## Sparse Transfer Learning with a Custom Dataset (Rotten Tomatoes)

Beyond the built-in GLUE tasks, we can also use a custom dataset from the Hugging Face Hub or from the local filesystem. 

Let's try an example with the [Rotten Tomatoes Dataset](https://huggingface.co/datasets/rotten_tomatoes), which cointains 5,331 positive and 5,331 negative sentences.

For simplicity, we will perform the fine-tuning without distillation. Although the transfer learning recipe contains distillation
modifiers, by setting `--distill_teacher disable` we instruct SparseML to skip distillation.

### Rotten Tomatoes Inspection

Run the following to inspect the Rotten Tomatoes dataset.

```python
from datasets import load_dataset

rotten_tomatoes = load_dataset("rotten_tomatoes")
print(rotten_tomatoes)
print(rotten_tomatoes["train"][0])

# {'text': 'the rock is destined to be the 21st century\'s new " conan " and that he\'s going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .', 'label': 1}
```

We can see that each row dataset contains a `text` field which is a string representing the sequence to be classified and a `label` field which is a `0` or `1` representing negative and positive labels.

### Using a Hugging Face Dataset Identifier

We can pass the Hugging Face dataset identifier to the CLI. Simply replace the `--task_name sst2` argument with `--dataset_name rotten_tomatoes --input_column_names text --label_column_name label`:

```
sparseml.transformers.text_classification \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --recipe zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none \
  --recipe_args '{"num_epochs":12,"qat_start_epoch":7.0, "observer_epoch": 11.0}' \
  --distill_teacher disable \
  --dataset_name rotten_tomatoes --input_column_names "text" --label_column_name "label" \
  --output_dir sparse_quantized_bert-text_classification_rotten_tomatoes-hf_dataset --max_seq_length 128 --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 --preprocessing_num_workers 6 --do_train --do_eval --evaluation_strategy epoch --fp16  \
  --save_strategy epoch --save_total_limit 1
```

You will notice that we used the same recipe as we did in the SST2 case (identified by the SparseZoo stub `zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none`). Since the Rotten Tomatoes and SST2 tasks are similiar, we chose to start with the same hyperparameters as we used in SST2 training. 

#### Recipe Args

To update a recipe and experiment with hyperparameters, you can download the YAML file from SparseZoo, make updates to the YAML directly, and pass the local path to SparseML.

Alternatively, you can use `--recipe_args` to modify the recipe on the fly. In this case, we used the following to run for 12 epochs instead of 13 (with QAT running over the final 5 epochs):

```bash
--recipe_args '{"num_epochs":12,"qat_start_epoch":7.0, "observer_epoch": 11.0}'
```

### Using Local CSV/JSON Files

Let's walk through how to pass a CSV/JSON dataset to the CLI.

#### Save Dataset as a CSV File

We use Hugging Face `datasets` to create a CSV file for Rotten Tomatoes that can be passed to SparseML's CLI:
```python
from datasets import load_dataset

rotten_tomatoes = load_dataset("rotten_tomatoes")
print(rotten_tomatoes)
print(rotten_tomatoes["train"][0])

rotten_tomatoes["train"].to_csv("./rotten_tomatoes-train.csv")
rotten_tomatoes["validation"].to_csv("./rotten_tomatoes-validation.csv")
```

We can see that the data is a CSV file with text and label as the columns:

```bash
head ./rotten_tomatoes-train.csv --lines=5
```

Output:
```
,text,label
0,"the rock is destined to be the 21st century's new "" conan "" and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .",1
1,"the gorgeously elaborate continuation of "" the lord of the rings "" trilogy is so huge that a column of words cannot adequately describe co-writer/director peter jackson's expanded vision of j . r . r . tolkien's middle-earth .",1
2,effective but too-tepid biopic,1
3,"if you sometimes like to go to the movies to have fun , wasabi is a good place to start .",1
```

#### **Kick off Training**

To use the local files with the CLI, pass `--train_file ./rotten_tomatoes-train.csv --validation_file ./rotten_tomatoes-validation.csv  --input_column_names text --label_column_name label`:

```bash
sparseml.transformers.text_classification \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --recipe zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none \
  --distill_teacher disable \
  --train_file ./rotten_tomatoes-train.csv --validation_file ./rotten_tomatoes-validation.csv  --input_column_names text --label_column_name label \
  --output_dir sparse_quantized_bert-text_classification_rotten_tomatoes-local_dataset --max_seq_length 128 --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 --preprocessing_num_workers 6 --do_train --do_eval --evaluation_strategy epoch --fp16  \
  --save_strategy epoch --save_total_limit 1
```

## Sparse Transfer Learning with a Custom Teacher (Rotten Tomatoes)

To increase accuracy, we can apply model distillation from a dense teacher model, just like we did for the SST2 case.
You are free to use the native Hugging Face workflows to train the dense teacher model (and can even
pass a Hugging Face model identifier to the `--distill_teacher` argument), but can also use the SparseML CLI.

### Train The Dense Teacher

Run the following to train a dense model on Rotten Tomatoes:
```
sparseml.transformers.train.text_classification \
  --output_dir dense-teacher_rotten_tomatoes \
  --model_name_or_path zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/base-none \
  --recipe zoo:nlp/sentiment_analysis/bert-base/pytorch/huggingface/sst2/base-none \
  --recipe_args '{"init_lr":0.00003}' \
  --dataset_name rotten_tomatoes --input_column_names "text" --label_column_name "label" \
  --max_seq_length 128 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --preprocessing_num_workers 6 \
  --do_train --do_eval --evaluation_strategy epoch --fp16  \
  --save_strategy epoch --save_total_limit 1
```

The model converges to ~86.9% accuracy without any hyperparameter search.

Note that used the dense version of BERT (the stub ends in `base-none`) as the starting point for the training 
and passed a recipe from [SparseZoo](https://sparsezoo.neuralmagic.com/models/nlp%2Fsentiment_analysis%2Fbert-base%2Fpytorch%2Fhuggingface%2Fsst2%2Fbase-none) which was used to train the 
dense teacher for the SST2 task. Since the SST2 task is similiar to the Rotten Tomatoes task, these parameters are a solid
place to start. This recipe contains no sparsity related modifiers and only controls the learning rate and number of epochs. As such, the script
will run typical fine-tuning, resulting in a dense model.

Here's what the recipe looks like:
```yaml
version: 1.1.0

# General Variables
num_epochs: 8
init_lr: 1.5e-4 
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

With the dense teacher trained, we can sparse transfer learn with the help of the teacher by passing a local path to the model checkpoint. In this case, we use `--distill_teacher ./dense-teacher_rotten_tomatoes`.

Run the following to kick-off training:

```bash
sparseml.transformers.text_classification \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --recipe zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none \
  --recipe_args '{"num_epochs":12,"qat_start_epoch":7.0, "observer_epoch": 11.0}' \
  --distill_teacher ./dense-teacher_rotten_tomatoes \
  --dataset_name rotten_tomatoes --input_column_names "text" --label_column_name "label" \
  --output_dir sparse_quantized_bert-text_classification_rotten_tomatoes-hf_dataset --max_seq_length 128 --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 --preprocessing_num_workers 6 --do_train --do_eval --evaluation_strategy epoch --fp16  \
  --save_strategy epoch --save_total_limit 1
```

The model converges to 85% accuracy.
