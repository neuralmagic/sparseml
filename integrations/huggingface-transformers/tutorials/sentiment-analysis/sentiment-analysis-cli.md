# Sentiment Analysis: Sparse Transfer Learning with the CLI

In this example, you will sparse transfer learn a 90% pruned BERT model onto some sentiment-analysis datasets using SparseML's CLI.

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

## Sparse Transfer Learning onto SST2 (GLUE Task)

SparseML's CLI enables you to kick-off sparsification workflows with various utilities like creating training pipelines, dataset loading, checkpoint saving, metric reporting, and logging handled for you. 

All we have to do is pass a couple of key arguments: 
- `--model_name_or_path` specifies the starting checkpoint to load for training
- `--task` specifies a glue dataset to train with 
- `--recipe` specifies path a recipe to use to apply sparsification algorithms or sparse transfer learning to the model. For Sparse Transfer Learning, we will use a recipe that instructs SparseML to maintain sparsity during the training process and to apply quantization over the final few epochs. 

### Run Transfer Learning

We will fine-tune a [90% pruned version of BERT](zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none) onto SST2.

Run the following:
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

- `--recipe zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none` specifies the recipe to be applied by SparseML. Here, we passed a SparseZoo stub identifying the transfer learning recipe for the SST2 dataset, which SparseML downloads when the script starts. See below for the details of what this recipe looks like.

- `--distill_teacher zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/base-none` is an optional argument that specifies a model to use for as a teacher to apply distillation during the training process. We passed a SparseZoo stub identifying a dense BERT model trained on SST2, which SparseML downloads when the script starts.

The model trains for 13 epochs, converging to ~92% accuracy on the validation set. Because we applied a sparse transfer recipe, which instructs SparseML to maintain the sparsity of the starting pruned checkpoint and apply quantization, the final model is 90% pruned and quantized!

#### Transfer Learning Recipe

SparseML's recipes are YAML files that specify the sparsity related algorithms and parameters. SparseML parses the recipes and updates the training loops to apply the 
to apply sparsification algorithms or sparse transfer learning to the model.

In the case of SST2, we used a [premade recipe from the SparseZoo](https://sparsezoo.neuralmagic.com/models/nlp%2Fsentiment_analysis%2Fobert-base%2Fpytorch%2Fhuggingface%2Fsst2%2Fpruned90_quant-none). 

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

</details>

You can download the recipe with the following code:

```python
from sparsezoo import Model
transfer_stub = "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none"
download_dir = "./transfer_recipe"
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
  --model_path sparse_quantized_bert-text_classification_sst2 \
  --task text_classification
```

A `deployment` folder is created in your local directory, which has all of the files needed for deployment with DeepSparse including the `model.onnx`, `config.json`, and `tokenizer.json` files.

## Sparse Transfer Learning with a Custom Dataset (Rotten Tomatoes)

Beyond the built-in GLUE tasks, we can also use a dataset from the Hugging Face Hub or pass via local files. Let's try an example of each for the sentiment analysis using [Rotten Tomatoes Dataset](https://huggingface.co/datasets/rotten_tomatoes), which containing 5,331 positive and 5,331 negative processed sentences.

For simplicity, we will perform the fine-tuning without distillation. Although the transfer learning recipe contains distillation
modifiers, by setting `--distill_teacher disable` we instruct SparseML to skip distillation.

### Rotten Tomatoes Inspection

Run the following to inspect the Rotten Tomatoes dataset.

```python
from datasets import load_dataset

rotten_tomatoes = load_dataset("rotten_tomatoes")
print(rotten_tomatoes)
print(rotten_tomatoes["train"][0])
```

We can see that each row dataset contains a `text` field which is a string representing the sequence to be classified and a `label` field which is a `0` or `1` representing
negative and positive labels.

### Using a Hugging Face Dataset Identifier

We can easily pass the `rotten_tomatoes` Hugging Face dataset identifier to train with this dataset instead of SST2. Simply replace the `--task_name sst2` argument with `--dataset_name rotten_tomatoes --input_column_names text --label_column_name label`. SparseML will then download the dataset from the Hugging Face hub and run training as before.

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

Alternatively, you can use `--recipe_args` to modify the recipe on the fly. In this case, we used the following to run for 12 epochs instead of 13:

```bash
--recipe_args '{"num_epochs":12,"qat_start_epoch":7.0, "observer_epoch": 11.0}'
```

### Using Local CSV/JSON Files

Let's walk through how to pass a CSV/JSON dataset to the CLI.

#### Save Dataset as a CSV File

For this example, we use Hugging Face `datasets` to create a CSV file for Rotten Tomatoes that can be passed to SparseML's CLI but you can use any framework you want to create the CSV.

Run the following to create the CSV files:

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

#### **Kick off Training**

To use the local files with the CLI, pass `--train_file ./rotten_tomatoes-train.csv --validation_file ./rotten_tomatoes-validation.csv  --input_column_names text --label_column_name label` in place 

Run the following:
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

### Sparse Transfer Learning with a Custom Teacher (Rotten Tomatoes)

To support the transfer learning process, we can apply model distillation, just like we did for the SST2 case.
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

The model converges to XX% accuracy.