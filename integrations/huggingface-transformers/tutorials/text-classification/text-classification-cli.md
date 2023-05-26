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

# Text Classification: Sparse Transfer Learning with the CLI

In this example, you will sparse transfer learn a 90% pruned BERT model onto some text classification datasets using SparseML's CLI.

### **Sparse Transfer Learning Overview**

Sparse Transfer Learning is very similiar to the typical transfer learning process used to train NLP models, where we fine-tune a pretrained checkpoint onto a smaller downstream dataset. With Sparse Transfer Learning, we simply start the fine-tuning process from a pre-sparsified checkpoint and maintain sparsity while the training process occurs.

### Pre-Sparsified BERT

SparseZoo, Neural Magic's open source repository of pre-sparsified models, contains a 90% pruned version of BERT, which has been sparsified on the upstream Wikipedia and BookCorpus datasets with the masked language modeling objective. We will use this model as the starting point for the transfer learning process.

- [Check out 90% pruned BERT model card](https://sparsezoo.neuralmagic.com/models/nlp%2Fmasked_language_modeling%2Fobert-base%2Fpytorch%2Fhuggingface%2Fwikipedia_bookcorpus%2Fpruned90-none)
- [Check out the full list of pre-sparsified NLP models](https://sparsezoo.neuralmagic.com/?domain=nlp&sub_domain=masked_language_modeling&page=1)

### Table of Contents

In this tutorial, you will learn how to:
- [Sparse Transfer Learn with a GLUE Dataset (Multi-Input Multi-Class - MNLI)](#sparse-transfer-learning-with-a-glue-dataset-multi-input-multi-class---mnli)
- [Sparse Transfer Learn with a GLUE Dataset (Multi-Input Binary-Class - QQP)](#sparse-transfer-learning-with-a-glue-dataset-multi-input-binary-class---qqp)
- [Sparse Transfer Learn with a Custom Dataset (Single-Input Multi-Class - TweetEval)](#sparse-transfer-learning-with-a-custom-dataset-single-input-multi-class---tweeteval)
- [Sparse Transfer Learn with a Custom Dataset (Multi-Input Multi-Class - SICK)](#sparse-transfer-learning-with-a-custom-dataset-multi-input-multi-class---sick)
- [Sparse Transfer Learn with a Custom Teacher (Singe-Input Binary-Class - Rotten Tomatoes)](#sparse-transfer-learning-with-a-custom-teacher-singe-input-binary-class---rotten-tomatoes)
- [Sparse Transfer Learn with a Custom Teacher from HF Hub (Singe-Input Multi-Class - TweetEval)](#sparse-transfer-learning-with-a-custom-teacher-from-hf-hub-singe-input-multi-class---tweeteval)

## Installation

Install SparseML via `pip`:

```bash
pip install sparseml[transformers]
```

## SparseML CLI

SparseML's CLI offers pre-made training pipelines for common NLP tasks, including text classification. 

The CLI enables you to kick-off training runs with various utilities like dataset loading and pre-processing, checkpoint saving, metric reporting, and logging handled for you.

All we have to do is pass a couple of key arguments: 
- `--model_name_or_path` specifies the starting checkpoint to load for training
- `--task` specifies a glue task to train on
- `--recipe` specifies path a recipe to use to apply sparsification algorithms or sparse transfer learning to the model. For Sparse Transfer Learning, we will use a recipe that instructs SparseML to maintain sparsity during the training process and to apply quantization over the final few epochs. 

Let's try some examples!

## Sparse Transfer Learning with a GLUE Dataset (Multi-Input Multi-Class - MNLI)

### MNLI Dataset

The Multi-Genre Natural Language Inference (MNLI) Corpus is a crowdsourced collection of sentence pairs with textual entailment annotations. Given a premise sentence and a hypothesis sentence, the task is to predict whether the premise entails the hypothesis (entailment), contradicts the hypothesis (contradiction), or neither (neutral). The premise sentences are gathered from ten different sources, including transcribed speech, fiction, and government reports. The authors of the benchmark use the standard test set, for which they obtained private labels from the RTE authors, and evaluate on both the matched (in-domain) and mismatched (cross-domain) section. They also uses and recommend the SNLI corpus as 550k examples of auxiliary training data.

[Check out the dataset card](https://huggingface.co/datasets/glue/viewer/mnli/train)

### Create a Transfer Learning Recipe

To launch a Sparse Transfer Learning run, we first need to create a Sparse Transfer Learning recipe.

Recipes are YAML files that specify sparsity related algorithms and hyper-parameters. SparseML parses the recipes and updates the training loops to apply the specified sparsification algorithms to the model.

In the case of MNLI, we used a [premade recipe from the SparseZoo](https://sparsezoo.neuralmagic.com/models/nlp%2Ftext_classification%2Fobert-base%2Fpytorch%2Fhuggingface%2Fmnli%2Fpruned90_quant-none) (shown here):

```yaml
version: 1.1.0

num_epochs: 13
init_lr: 8e-5
final_lr: 0

qat_start_epoch: 8.0
observer_epoch: 12.0
quantize_embeddings: 1

distill_hardness: &distill_hardness 1.0
distill_temperature: &distill_temperature 3.0

weight_decay: 0.0

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

SparseML parses the modifiers and updates the training process to implement the algorithms and hyperparameters specified in the recipes. As such, when this recipe is passed, the sparsity structure of the network will be maintained while the fine-tuning occurs and the weights will be quantized over the final few epochs.

You can download the recipe with the following code:

```python
from sparsezoo import Model
transfer_stub = "zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none"
download_dir = "./transfer_recipe-mnli"
zoo_model = Model(transfer_stub, download_path=download_dir)
recipe_path = zoo_model.recipes.default.path
print(recipe_path)
```

### Fine Tune The Model

With the recipe and starting sparse checkpoint identified, we can kick off the fine-tuning with the following:
```bash
sparseml.transformers.text_classification \
  --task_name mnli \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --recipe zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none \
  --distill_teacher zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/base-none \
  --output_dir sparse_quantized_bert-text_classification_mnli \
  --do_train --do_eval --max_seq_length 128 --evaluation_strategy epoch --logging_steps 1000 --save_steps 1000 \
  --per_device_train_batch_size 8 --per_device_eval_batch_size 32 --gradient_accumulation_steps 4 --preprocessing_num_workers 32 \
  --seed 5114
```

Let's discuss the key arguments:
- `--task_name mnli` instructs SparseML to download and fine-tune onto the MNLI dataset. You can pass any GLUE task to this parameter.

- `--model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none` specifies the starting checkpoint for the fine tuning. Here, we passed a SparseZoo stub identifying the 90% pruned version of BERT trained with masked language modeling, which SparseML downloads when the script starts.

- `--recipe zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none` specifies the recipe to be applied by SparseML. Here, we passed a SparseZoo stub identifying the transfer learning recipe for the MNLI dataset, which SparseML downloads when the script starts.

- `--distill_teacher zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/base-none` is an optional argument that specifies a model to use for as a teacher to apply distillation during the training process. We passed a SparseZoo stub identifying a dense BERT model trained on MNLI, which SparseML downloads when the script starts.

The model trains for 13 epochs, converging to ~82.5% accuracy on the validation set. Because we applied a sparse transfer recipe, which instructs SparseML to maintain the sparsity of the starting pruned checkpoint and apply quantization, the final model is 90% pruned and quantized!

### **Export to ONNX**

Export your trained model to ONNX to deploy with DeepSparse:

```bash
sparseml.transformers.export_onnx \
  --model_path sparse_quantized_bert-text_classification_mnli \
  --task text_classification
```

A `deployment` folder is created in your local directory, which has all of the files needed for deployment with DeepSparse including the `model.onnx`, `config.json`, and `tokenizer.json` files.

## Sparse Transfer Learning with a GLUE Dataset (Multi-Input Binary-Class - QQP)

### QQP Dataset

The Quora Question Pairs2 dataset is a collection of question pairs from the community question-answering website Quora. The task is to determine whether a pair of questions are semantically equivalent.

[Check out the dataset card](https://huggingface.co/datasets/glue/viewer/qqp/test)

### Create a Transfer Learning Recipe

As with MNLI, we first need a transfer learning recipe.

In the case of QQP, there is a [premade recipe available in the SparseZoo](https://sparsezoo.neuralmagic.com/models/nlp%2Ftext_classification%2Fobert-base%2Fpytorch%2Fhuggingface%2Fqqp%2Fpruned90_quant-none):

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
```


The `Modifiers` are the important items that encode how SparseML should modify the training process for Sparse Transfer Learning:
- `ConstantPruningModifier` tells SparseML to pin weights at 0 over all epochs, maintaining the sparsity structure of the network
- `QuantizationModifier` tells SparseML to quanitze the weights with quantization aware training over the last 5 epochs
- `DistillationModifier` tells SparseML how to apply distillation during the trainign process, targeting the logits

SparseML parses the modifiers and updates the training process to implement the algorithms and hyperparameters specified in the recipes. As such, when this recipe is passed, the sparsity structure of the network will be maintained while the fine-tuning occurs and the weights will be quantized over the final few epochs.

You can download the recipe with the following code:

```python
from sparsezoo import Model
transfer_stub = "zoo:nlp/text_classification/obert-base/pytorch/huggingface/qqp/pruned90_quant-none"
download_dir = "./transfer_recipe-qqp"
zoo_model = Model(transfer_stub, download_path=download_dir)
recipe_path = zoo_model.recipes.default.path
print(recipe_path)
```

### Run Transfer Learning

With the recipe created and starting model identified, we can swap `--task_name mnli` for `--task_name qqp` along with the appropriate `recipe` and `distill_teacher` model stubs.

Run the following to fine-tune the [90% pruned version of BERT](zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none) onto QQP:
```bash
sparseml.transformers.text_classification \
  --task_name qqp \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --recipe zoo:nlp/text_classification/obert-base/pytorch/huggingface/qqp/pruned90_quant-none \
  --distill_teacher zoo:nlp/text_classification/obert-base/pytorch/huggingface/qqp/base-none \
  --output_dir obert_base_pruned90_quant_qqp \
  --do_train --do_eval --evaluation_strategy epoch --logging_steps 1000 --save_steps 1000 \
  --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --preprocessing_num_workers 32 \
  --max_seq_length 128 \
  --seed 10194
```

Let's discuss the key arguments:
- `--task_name qqp` instructs SparseML to download and fine-tune onto the QQP dataset. You can pass any GLUE task to this parameter.

- `--model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none` specifies the starting checkpoint for the fine tuning. Here, we passed a SparseZoo stub identifying the 90% pruned version of BERT trained with masked language modeling, which SparseML downloads when the script starts.

- `--recipe zoo:nlp/text_classification/obert-base/pytorch/huggingface/qqp/pruned90_quant-none` specifies the recipe to be applied by SparseML. Here, we passed a SparseZoo stub identifying the transfer learning recipe for the QQP dataset, which SparseML downloads when the script starts.

- `--distill_teacher zoo:nlp/text_classification/obert-base/pytorch/huggingface/qqp/base-none` is an optional argument that specifies a model to use for as a teacher to apply distillation during the training process. We passed a SparseZoo stub identifying a dense BERT model trained on QQP, which SparseML downloads when the script starts.

The model trains for 13 epochs, converging to ~91% accuracy on the validation set. Because we applied a sparse transfer recipe, which instructs SparseML to maintain the sparsity of the starting pruned checkpoint and apply quantization, the final model is 90% pruned and quantized!

## Sparse Transfer Learning with a Custom Dataset (Single-Input Multi-Class - TweetEval)

Beyond the built-in GLUE tasks, we can also use custom text classification datasets. The datasets can either be passed as Hugging Face hub dataset identifiers or via local CSV files.

Let's try to transfer onto the [TweetEval Emotion Dataset](https://huggingface.co/datasets/tweet_eval). This dataset 
contains single sentences with 4 labels representing the emotion of the tweet (`0=anger, 1=joy, 2=optimism, 3=sadness`).

For simplicity, we will perform the fine-tuning without distillation. Although the transfer learning recipe contains distillation modifiers, we can turn them off by setting `--distill_teacher disable`.

### Inspecting TweetEval Dataset

Run the following to inspect the TweetEval dataset:

```python
from datasets import load_dataset
from pprint import pprint

emotion = load_dataset("tweet_eval", "emotion")
print(emotion)
pprint(emotion["train"][0])

# > {'label': 2,
# >  'text': "“Worry is a down payment on a problem you may never have'. \xa0Joyce "'Meyer.  #motivation #leadership #worry'}
```

We can see that each row dataset contains a `text` field which is a string representing the sequence to be classified and a `label` field which is in `{0,1,2,3}` representing
one of four emotions.

### Using a Hugging Face Dataset Identifier

We can pass the Hugging Face dataset identifier to the CLI with `--dataset_name tweet_eval --dataset_config_name emotion --input_column_names text --label_column_name label`. 

Since the TweetEval dataset contains multiple subsets (e.g. there is a subset that classifies text into an emoji), we pass the `--dataset_config_name` to specify the `emotion` subset. SparseML will then download the dataset from the Hugging Face hub and run training as before.

Run the following to kick off the training process:
```
sparseml.transformers.text_classification \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --recipe zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none \
  --recipe_args '{"num_epochs":12,"qat_start_epoch":7.0, "observer_epoch": 11.0}' \
  --distill_teacher disable \
  --dataset_name tweet_eval --dataset_config_name emotion \
  --input_column_names "text" --label_column_name "label" \
  --output_dir sparse_quantized_bert-text_classification_tweet_eval_emotion \
  --do_train --do_eval --max_seq_length 128 --evaluation_strategy epoch --logging_steps 1000 --save_steps 1000 \
  --per_device_train_batch_size 8 --per_device_eval_batch_size 32 --gradient_accumulation_steps 4 --preprocessing_num_workers 32 \
  --seed 5114
```

You will notice that we used the same recipe as we did in the SST2 case (identified by the SparseZoo stub `zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none`). Since the TweetEval Emotion dataset is a single sentence multi-class classification problem, we used the transfer learning recipe from the sentiment analysis task (a single sentence binary classification problem) as the starting point.

Let's see how we can experiment with some of the hyperparameters.

#### Recipe Args

To update a recipe and experiment with hyperparameters, you can download the YAML file from SparseZoo, make updates to the YAML directly, and pass the local path to SparseML. Alternatively, you can use `--recipe_args` to modify the recipe on the fly. In this case, we used the following to run for 12 epochs instead of 13 (with QAT over the final 5 epochs):

```bash
--recipe_args '{"num_epochs":12,"qat_start_epoch":7.0, "observer_epoch": 11.0}'
```

### Using Local CSV/JSON Files

Let's walk through how to pass a CSV/JSON dataset to the CLI.

#### Save Dataset as a CSV File

For this example, we use Hugging Face `datasets` to create a CSV file for TweetEval  that can be passed to SparseML's CLI:

```python
from datasets import load_dataset

emotion = load_dataset("tweet_eval", "emotion")
print(emotion)
print(emotion["train"][0])
emotion["train"].to_csv("./emotion-train.csv")
emotion["validation"].to_csv("./emotion-validation.csv")
```

We can see that the data is a CSV file with text and label as the columns:

```bash
head ./emotion-train.csv --lines=5
```

Output:
```bash
,text,label
0,“Worry is a down payment on a problem you may never have'.  Joyce Meyer.  #motivation #leadership #worry,2
1,My roommate: it's okay that we can't spell because we have autocorrect. #terrible #firstworldprobs,0
2,No but that's so cute. Atsu was probably shy about photos before but cherry helped her out uwu,1
3,"Rooneys fucking untouchable isn't he? Been fucking dreadful again, depay has looked decent(ish)tonight",0
```

#### **Kick off Training**

To use the local files with the CLI, pass `--train_file ./emotion-train.csv --validation_file ./emotion-validation.csv  --input_column_names text --label_column_name label`:

```bash
sparseml.transformers.text_classification \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --recipe zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none \
  --recipe_args '{"num_epochs":12,"qat_start_epoch":7.0, "observer_epoch": 11.0}' \
  --distill_teacher disable \
  --train_file ./emotion-train.csv --validation_file ./emotion-validation.csv \
  --input_column_names "text" --label_column_name "label" \
  --output_dir sparse_quantized_bert-text_classification_tweet_eval_emotion-csv \
  --do_train --do_eval --max_seq_length 128 --evaluation_strategy epoch --logging_steps 1000 --save_steps 1000 \
  --per_device_train_batch_size 8 --per_device_eval_batch_size 32 --gradient_accumulation_steps 4 --preprocessing_num_workers 32 \
  --seed 5114
```

## Sparse Transfer Learning with a Custom Dataset (Multi-Input Multi-Class - SICK)

Let's try to transfer onto the [SICK (Sentences Involving Compositional Knowledge) Dataset](https://huggingface.co/datasets/sick),
which includes 10,000 pairs of sentences with entailment relationships (`0=entailment, 1=neural, 2=contradiction`).

For simplicity, we will perform the fine-tuning without distillation. Although the transfer learning recipe contains distillation modifiers, we can turn them off by setting `--distill_teacher disable`.

### Inspecting SICK Dataset

Run the following to inspect the TweetEval

```python
from datasets import load_dataset
from pprint import pprint

sick = load_dataset("sick")
print(sick)
pprint(sick["train"][0])

# > {'label': 1,
# >  'sentence_A': 'A group of kids is playing in a yard and an old man is standing in the background',
# >  'sentence_B': 'A group of boys in a yard is playing and a man is standing in the background'}
```

We can see that each row dataset contains two input fields (`sentence_A` and `sentence_B`) which are strings representing the sequence pairs to be classified and a `label` field which is in `{0,1,2}` representing the entailment relationship.

### Using a Hugging Face Dataset Identifier

Pass the `sick` Hugging Face dataset identifier to the script with `--dataset_name sick --input_column_names 'sentence_A,sentence_B' --label_column_name label`:

```bash
sparseml.transformers.text_classification \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --recipe zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none \
  --distill_teacher disable \
  --dataset_name sick --input_column_names 'sentence_A,sentence_B' --label_column_name 'label' \
  --output_dir sparse_quantized_bert-text_classification_sick \
  --do_train --do_eval --max_seq_length 128 --evaluation_strategy epoch --logging_steps 1000 --save_steps 1000 \
  --per_device_train_batch_size 8 --per_device_eval_batch_size 32 --gradient_accumulation_steps 4 --preprocessing_num_workers 32 \
  --seed 5114
```

You will notice that we used the same recipe as we did in the MNLI case (identified by the SparseZoo stub `zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none`). 

Since the MNLI dataset is a multi sentence multi-class classification problem (similiarly, it is an entailment problem), the hyperparameters used for MNLI are a solid starting point for SICK.

To experiment with the hyperparameters, you can download the YAML file from SparseZoo, make updates to the YAML directly, and pass the local path to SparseML. Alternatively, you can use `--recipe_args` to update on the fly.

### Using Local CSV Files

Let's walk through how to pass a CSV dataset to the CLI.

#### Save Dataset as a CSV File

We use Hugging Face `datasets` to create a CSV file:

```python
from datasets import load_dataset

sick = load_dataset("sick")
print(sick)
print(sick["train"][0])
sick["train"].to_csv("./sick-train.csv")
sick["validation"].to_csv("./sick-validation.csv")
```

We can see that the data is a CSV file with `sentence_A` and `sentence_B` as the input columns and `label` as the label column:

```bash
head ./sick-train.csv --lines=5
```

Output:

```bash
,id,sentence_A,sentence_B,label,relatedness_score,entailment_AB,entailment_BA,sentence_A_original,sentence_B_original,sentence_A_dataset,sentence_B_dataset
0,1,A group of kids is playing in a yard and an old man is standing in the background,A group of boys in a yard is playing and a man is standing in the background,1,4.5,A_neutral_B,B_neutral_A,"A group of children playing in a yard, a man in the background.","A group of children playing in a yard, a man in the background.",FLICKR,FLICKR
1,2,A group of children is playing in the house and there is no man standing in the background,A group of kids is playing in a yard and an old man is standing in the background,1,3.2,A_contradicts_B,B_neutral_A,"A group of children playing in a yard, a man in the background.","A group of children playing in a yard, a man in the background.",FLICKR,FLICKR
2,3,The young boys are playing outdoors and the man is smiling nearby,The kids are playing outdoors near a man with a smile,0,4.7,A_entails_B,B_entails_A,"The children are playing outdoors, while a man smiles nearby.","The children are playing outdoors, while a man smiles nearby.",FLICKR,FLICKR
3,5,The kids are playing outdoors near a man with a smile,A group of kids is playing in a yard and an old man is standing in the background,1,3.4,A_neutral_B,B_neutral_A,"A group of children playing in a yard, a man in the background.","The children are playing outdoors, while a man smiles nearby.",FLICKR,FLICKR
```

#### Kick off Training

To use the local files with the CLI, pass `--train_file ./sick-train.csv --validation_file ./sick-validation.csv  --input_column_names 'sentence_A,sentence_B' --label_column_name label`.

Run the following:
```bash
sparseml.transformers.text_classification \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --recipe zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none \
  --distill_teacher disable \
  --train_file ./sick-train.csv --validation_file ./sick-validation.csv --input_column_names 'sentence_A,sentence_B' --label_column_name 'label' \
  --output_dir sparse_quantized_bert-text_classification_sick-csv \
  --do_train --do_eval --max_seq_length 128 --evaluation_strategy epoch --logging_steps 1000 --save_steps 1000 \
  --per_device_train_batch_size 8 --per_device_eval_batch_size 32 --gradient_accumulation_steps 4 --preprocessing_num_workers 32 \
  --seed 5114
```

## Sparse Transfer Learning with a Custom Teacher (Singe-Input Binary-Class - Rotten Tomatoes)

To support the transfer learning process, we can apply model distillation, just like we did for the GLUE task.
You are free to use the native Hugging Face workflows to train the dense teacher model (and can even
pass a Hugging Face model identifier to the command), but you can also use the SparseML CLI for the dense training.

### Train The Dense Teacher

Run the following to train a dense model on Rotten Tomatoes:
```
sparseml.transformers.text_classification \
  --output_dir dense-teacher_rotten_tomatoes \
  --model_name_or_path zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/base-none \
  --recipe zoo:nlp/sentiment_analysis/bert-base/pytorch/huggingface/sst2/base-none \
  --recipe_args '{"init_lr":0.00003}' \
  --dataset_name rotten_tomatoes --input_column_names "text" --label_column_name "label" \
  --max_seq_length 128 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --preprocessing_num_workers 6 \
  --do_train --do_eval --evaluation_strategy epoch --fp16  \
  --save_strategy epoch --save_total_limit 1
```

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

### Sparse Transfer Learning with a Custom Teacher

With the dense teacher trained, we can sparse transfer learn with the help of the teacher by passing a local path to the model checkpoint (in this case, 
`--distill_teacher ./dense-teacher_rotten_tomatoes`).

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

## Sparse Transfer Learning with a Custom Teacher from HF Hub (Singe-Input Multi-Class - TweetEval)

In addition to passing models trained locally, we can also pass models from the Hugging 
Face hub via model identifiers to the transfer learning script as the `--distill_teacher`.

For example, we can pass the [RoBERTa-base](https://huggingface.co/cardiffnlp/roberta-base-emotion) model from the Hugging Face hub that has been fine-tuned on the TweetEval emotion dataset as the teacher by passing the Hugging Face model identifier `cardiffnlp/twitter-roberta-base-emotion`

Run the following:
```
sparseml.transformers.text_classification \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --recipe zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none \
  --distill_teacher cardiffnlp/twitter-roberta-base-emotion \
  --use_teacher_tokenizer true \
  --dataset_name tweet_eval --dataset_config_name emotion \
  --input_column_names "text" --label_column_name "label" \
  --output_dir sparse_quantized_bert-text_classification_tweet_eval_emotion-hf_hub_teacher \
  --do_train --do_eval --max_seq_length 128 --evaluation_strategy epoch --logging_steps 1000 --save_steps 1000 \
  --per_device_train_batch_size 8 --per_device_eval_batch_size 32 --gradient_accumulation_steps 4 --preprocessing_num_workers 32 \
  --seed 5114
```

The model converges to 78% accuracy on the validaiton set.

Note that we had to pass `--use_teacher_tokenizer true` because RoBERTa and BERT use different tokenizers. This instructs SparseML to use different tokenizers for each the teacher and student.
