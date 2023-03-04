# Text Classification: Sparse Transfer Learning with the CLI

In this example, you will sparse transfer learn a 90% pruned BERT model onto some text classification datasets using SparseML's CLI.

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

## Sparse Transfer Learning onto MNLI (GLUE Task)

SparseML's CLI enables you to kick-off sparsification workflows with various utilities like creating training pipelines, dataset loading, checkpoint saving, metric reporting, and logging handled for you. 

All we have to do is pass a couple of key arguments: 
- `--model_name_or_path` specifies the starting checkpoint to load for training
- `--task` specifies a glue dataset to train with 
- `--recipe` specifies path a recipe to use to apply sparsification algorithms or sparse transfer learning to the model. For Sparse Transfer Learning, we will use a recipe that instructs SparseML to maintain sparsity during the training process and to apply quantization over the final few epochs. 

### Run Transfer Learning

We will fine-tune a [90% pruned version of BERT](zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none) onto MNLI.

Run the following:
```bash
sparseml.transformers.train.text_classification \
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

- `--recipe zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none` specifies the recipe to be applied by SparseML. Here, we passed a SparseZoo stub identifying the transfer learning recipe for the MNLI dataset, which SparseML downloads when the script starts. See below for the details of what this recipe looks like.

- `--distill_teacher zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/base-none` is an optional argument that specifies a model to use for as a teacher to apply distillation during the training process. We passed a SparseZoo stub identifying a dense BERT model trained on MNLI, which SparseML downloads when the script starts.

The model trains for 12 epochs, converging to ~82.5% accuracy on the validation set. Because we applied a sparse transfer recipe, which instructs SparseML to maintain the sparsity of the starting pruned checkpoint and apply quantization, the final model is 90% pruned and quantized!

#### Transfer Learning Recipe

SparseML's recipes are YAML files that specify the sparsity related algorithms and parameters. SparseML parses the recipes and updates the training loops to apply the 
to apply sparsification algorithms or sparse transfer learning to the model.

In the case of MNLI, we used a [premade recipe from the SparseZoo](https://sparsezoo.neuralmagic.com/models/nlp%2Ftext_classification%2Fobert-base%2Fpytorch%2Fhuggingface%2Fmnli%2Fpruned90_quant-none). 

<details>
    <summary>Click to inspect the recipe</summary>

The `Modifiers` are the important items that encode how SparseML should modify the training process for Sparse Transfer Learning:
- `ConstantPruningModifier` tells SparseML to pin weights at 0 over all epochs, maintaining the sparsity structure of the network
- `QuantizationModifier` tells SparseML to quanitze the weights with quantization aware training over the last 5 epochs
- `DistillationModifier` tells SparseML how to apply distillation during the trainign process, targeting the logits

SparseML parses the modifiers and updates the training process to implement the algorithms and hyperparameters specified in the recipes.

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

</details>

You can download the recipe with the following code:

```python
from sparsezoo import Model
transfer_stub = "zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none"
download_dir = "./transfer_recipe-mnli"
zoo_model = Model(transfer_stub, download_path=download_dir)
recipe_path = zoo_model.recipes.default.path
print(recipe_path)
```

### **Export to ONNX**

Once you have trained your model, export to ONNX in order to deploy with DeepSparse. The artifacts of the training process 
are saved to your local filesystem. 

Run the following to convert your PyTorch checkpoint to ONNX:

```bash
sparseml.transformers.export_onnx \
  --model_path sparse_quantized_bert-text_classification_mnli \
  --task text_classification
```

A `deployment` folder is created in your local directory, which has all of the files needed for deployment with DeepSparse including the `model.onnx`, `config.json`, and `tokenizer.json` files.

## Sparse Transfer Learning onto QQP (GLUE Task)

To train with the `QQP` dataset instead of `MNLI`, we can swap `--task_name mnli` for 
`--task_name qqp` with the appropriate `recipe` and `distill_teacher`

### Run Transfer Learning

We will fine-tune a [90% pruned version of BERT](zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none) onto QQP.

Run the following:
```bash
sparseml.transformers.train.text_classification \
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

- `--recipe zoo:nlp/text_classification/obert-base/pytorch/huggingface/qqp/pruned90_quant-none` specifies the recipe to be applied by SparseML. Here, we passed a SparseZoo stub identifying the transfer learning recipe for the QQP dataset, which SparseML downloads when the script starts. See below for the details of what this recipe looks like.

- `--distill_teacher zoo:nlp/text_classification/obert-base/pytorch/huggingface/qqp/base-none` is an optional argument that specifies a model to use for as a teacher to apply distillation during the training process. We passed a SparseZoo stub identifying a dense BERT model trained on QQP, which SparseML downloads when the script starts.

The model trains for 3 epochs, converging to ~91% accuracy on the validation set. Because we applied a sparse transfer recipe, which instructs SparseML to maintain the sparsity of the starting pruned checkpoint and apply quantization, the final model is 90% pruned and quantized!

#### Transfer Learning Recipe

SparseML's recipes are YAML files that specify the sparsity related algorithms and parameters. SparseML parses the recipes and updates the training loops to apply the 
to apply sparsification algorithms or sparse transfer learning to the model.

In the case of QQP, we used a [premade recipe from the SparseZoo](https://sparsezoo.neuralmagic.com/models/nlp%2Ftext_classification%2Fobert-base%2Fpytorch%2Fhuggingface%2Fqqp%2Fpruned90_quant-none). 

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

</details>

You can download the recipe with the following code:

```python
from sparsezoo import Model
transfer_stub = "zoo:nlp/text_classification/obert-base/pytorch/huggingface/qqp/pruned90_quant-none"
download_dir = "./transfer_recipe-qqp"
zoo_model = Model(transfer_stub, download_path=download_dir)
recipe_path = zoo_model.recipes.default.path
print(recipe_path)
```

### **Export to ONNX**

Once you have trained your model, export to ONNX in order to deploy with DeepSparse. The artifacts of the training process 
are saved to your local filesystem. 

Run the following to convert your PyTorch checkpoint to ONNX:

```bash
sparseml.transformers.export_onnx \
  --model_path obert_base_pruned90_quant_qqp \
  --task text_classification
```

A `deployment` folder is created in your local directory, which has all of the files needed for deployment with DeepSparse including the `model.onnx`, `config.json`, and `tokenizer.json` files.

## Sparse Transfer Learning with a Single Input Custom Dataset (TweetEval)

Beyond the built-in GLUE tasks, we can also use datasets for single input multi-class classification problems. The datasets can either be passed as Hugging Face hub model identifiers or via local CSV/JSON files.

Let's try to transfer onto the [TweetEval Emotion Dataset](https://huggingface.co/datasets/tweet_eval). This dataset 
contains single sentences with 4 labels representing the emotion of the tweet (`0=anger, 1=joy, 2=optimism, 3=sadness`.

For simplicity, we will perform the fine-tuning without distillation. Although the transfer learning recipe contains distillation modifiers, we can turn them off by setting `--distill_teacher disable`.

### Inspecting TweetEval Dataset

Run the following to inspect the TweetEval

```python
from datasets import load_dataset
from pprint import pprint

emotion = load_dataset("tweet_eval", "emotion")
print(emotion)
pprint(emotion["train"][0])
```

We can see that each row dataset contains a `text` field which is a string representing the sequence to be classified and a `label` field which is in `{0,1,2,3}` representing
one of four emotions.

### Using a Hugging Face Dataset Identifier

We can easily pass the `tweet_eval` Hugging Face dataset identifier to the script. Simply replace the `--task_name` argument with `--dataset_name tweet_eval --dataset_config_name emotion --input_column_names text --label_column_name label`. Since the TweetEval dataset contains multiple subsets (e.g. there is a subset
that classifies text into an emoji), we pass the `--dataset_config_name` to specify the subset. SparseML will then download the dataset from the Hugging Face hub and run training as before.

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

#### Recipe Args

To update a recipe and experiment with hyperparameters, you can download the YAML file from SparseZoo, make updates to the YAML directly, and pass the local path to SparseML.

Alternatively, you can use `--recipe_args` to modify the recipe on the fly. In this case, we used the following to run for 12 epochs instead of 13:

```bash
--recipe_args '{"num_epochs":12,"qat_start_epoch":7.0, "observer_epoch": 11.0}'
```

### Using Local CSV/JSON Files

Let's walk through how to pass a CSV/JSON dataset to the CLI.

#### Save Dataset as a CSV File

For this example, we use Hugging Face `datasets` to create a CSV file for TweetEval  that can be passed to SparseML's CLI but you can use any framework you want to create the CSV.

Run the following to create the CSV files:

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

#### **Kick off Training**

To use the local files with the CLI, pass `--train_file ./emotion-train.csv --validation_file ./emotion-validation.csv  --input_column_names text --label_column_name label`.

Run the following:
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

## Sparse Transfer Learning with a Multi Input Custom Dataset (SICK)

Beyond the built-in GLUE tasks, we can also use datasets for multiple input multi-class classification problems. The datasets can either be passed as Hugging Face hub model identifiers or via local CSV/JSON files.

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
```

We can see that each row dataset contains two input fields (`sentence_A` and `sentence_B`) which are strings representing the sequence pairs to be classified and a `label` field which is in `{0,1,2}` representing the entailment relationship.

### Using a Hugging Face Dataset Identifier

We can easily pass the `sick` Hugging Face dataset identifier to the script. Simply replace the `--task_name` argument with `--dataset_name sick --input_column_names 'sentence_A,sentence_B' --label_column_name label`.

Run the following to kick off the training process:
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

Since the MNLI dataset is a multi sentence multi-class classification problem (similiarly, it is an entailment problem), we used the transfer learning recipe from the sentiment analysis task (a single sentence binary classification problem) as the starting point.

To update a recipe, you can download the YAML file from SparseZoo, make updates to the YAML directly, and pass the local path to SparseML. Alternative, you can use `--recipe_args` to update on the fly.

### Using Local CSV/JSON Files

Let's walk through how to pass a CSV/JSON dataset to the CLI.

#### Save Dataset as a CSV File

For this example, we use Hugging Face `datasets` to create a CSV file for the SICK dataset that can be passed to SparseML's CLI but you can use any framework you want to create the CSV.

Run the following to create the CSV files:

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

#### **Kick off Training**

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

### Sparse Transfer Learning with a Custom Teacher (Rotten Tomatoes)

To support the transfer learning process, we can apply model distillation, just like we did for the GLUE task case.
You are free to use the native Hugging Face workflows to train the dense teacher model (and can even
pass a Hugging Face model identifier to the command), but you can also use the SparseML CLI as well. 

#### Train The Dense Teacher

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

#### Sparse Transfer Learning with a Custom Teacher

With the dense teacher trained, we can sparse transfer learn with the help of the teacher by passing
`--distill_teacher ./dense-teacher_rotten_tomatoes`.

Run the following to kick-off training:

```bash
sparseml.transformers.text_classification \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --recipe zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none \
  --recipe_args '{"num_epochs":12,"qat_start_epoch":7.0, "observer_epoch": 11.0}' \
  --distill_teacher --distill_teacher ./dense-teacher_rotten_tomatoes \
  --dataset_name rotten_tomatoes --input_column_names "text" --label_column_name "label" \
  --output_dir sparse_quantized_bert-text_classification_rotten_tomatoes-hf_dataset --max_seq_length 128 --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 --preprocessing_num_workers 6 --do_train --do_eval --evaluation_strategy epoch --fp16  \
  --save_strategy epoch --save_total_limit 1
```

### Sparse Transfer Learning with a Custom Teacher from HF Hub (TweetEval)

In addition to passing models trained locally, we can also pass models from the Hugging 
Face hub via model identifiers to the transfer learning script as the `--distill_teacher`.

For example, we can pass the [roBERTa-base](https://huggingface.co/cardiffnlp/roberta-base-emotion) model from the Hugging Face hub that has been fine-tuned on the TweetEval emotion dataset as the teacher by passing the Hugging Face model identifier `cardiffnlp/twitter-roberta-base-emotion`

Run the following:
```
sparseml.transformers.text_classification \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --recipe zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none \
  --distill_teacher cardiffnlp/twitter-roberta-base-emotion \
  --use_teacher_tokenizer true \
  --dataset_name tweet_eval --dataset_config_name emotion \
  --input_column_names "text" --label_column_name "label" \
  --output_dir sparse_quantized_bert-text_classification_tweet_eval_emotion \
  --do_train --do_eval --max_seq_length 128 --evaluation_strategy epoch --logging_steps 1000 --save_steps 1000 \
  --per_device_train_batch_size 8 --per_device_eval_batch_size 32 --gradient_accumulation_steps 4 --preprocessing_num_workers 32 \
  --seed 5114
```

The model converges to 78% accuracy on the validaiton set.

Note that we had to pass `--use_teacher_tokenizer true` because RoBERTa and BERT use different tokenizers. This instructs SparseML to use different tokenizers for each the twacher and student.
