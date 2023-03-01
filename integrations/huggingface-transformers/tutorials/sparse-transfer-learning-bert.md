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

# Sparse Transfer Learning

This page explains how to fine-tune a pre-sparsified BERT model onto a downstream dataset with SparseML.

## Overview

Sparse Transfer is quite similiar to the typical NLP transfer learning, where we fine-tune a checkpoint pretrained on a large dataset like Wikipedia BookCorpus onto a smaller downstream dataset and task. However, with Sparse Transfer Learning, we simply start the fine-tuning process from a pre-sparsified model and maintain sparsity while the training process occurs.

[SparseZoo](https://sparsezoo.neuralmagic.com/?domain=nlp&sub_domain=masked_language_modeling&page=1), Neural Magic's open-source Model Zoo, contains pre-sparsified checkpoints of common NLP models like BERT-base, BERT-large, and RoBERTa. These models can be used as the starting checkpoint for the sparse transfer learning workflow.

## Installation

Install via `pip`:

```
pip install sparseml[torch]
```

## Sparse Transfer Learning onto SST2

Let's try a simple example of fine-tuning a pre-sparsified model onto the SST dataset. SST2 is a sentiment analysis
dataset, with pairs of sentences and binary labels representing positive or negative sentiment.

[SST2 Dataset Card](https://huggingface.co/datasets/glue/viewer/sst2/train)

### Selecting a Pre-sparsified Upstream Model

We will fine-tune a [90% sparsified BERT-base](https://arxiv.org/abs/2203.07259). This model is hosted in [SparseZoo](https://sparsezoo.neuralmagic.com/models/nlp%2Fmasked_language_modeling%2Fobert-base%2Fpytorch%2Fhuggingface%2Fwikipedia_bookcorpus%2Fpruned90-none) and is identified by the following stub:
```bash
zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none
```

SparseML uses the stub to download the model before starting training.

### Kick off Training

Run the following to fine-tune the pre-sparsified BERT onto SST2:

```bash
sparseml.transformers.text_classification \
  --output_dir pruned_quantized_obert-text_classification_sst2 \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --recipe zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none \
  --distill_teacher zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/base-none \
  --task_name sst2 --max_seq_length 128 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --preprocessing_num_workers 6 \
  --do_train --do_eval --evaluation_strategy epoch --fp16  \
  --save_strategy epoch --save_total_limit 1
```

Lets discuss the key arguments:

- `--task` specifies the dataset to train on (in this case SST2). You can pass any GLUE task to the `--task` command. Check out the use case pages for details on passing a custom dataset.

- `--model_name_or_path` specifies the starting checkpoint for the training process. Here, we passed a SparseZoo stub, which
identifies the 90% pruned BERT model in the SparseZoo. The script downloads the PyTorch model to begin training. In addition to SparseZoo stubs, you can also pass a local path to a PyTorch checkpoint.

- `--recipe` specifies the transfer learning recipe. Recipes are YAML files that declare the sparsity related algorithms
 that SparseML should apply. For transfer learning, the recipe instructs SparseML to maintain sparsity during training
 and to apply quantization over the final epochs. In this case, we passed a SparseZoo stub, which instructs SparseML
 to download a premade SST2 transfer learning recipe. In addition to SparseZoo stubs, you can also pass a local path to a YAML recipe. 
 See below for more details on what the transfer learning recipe looks like.

- `--distill_teacher` is an optional argument that allows you to apply model distillation during fine-tuning. Here, we pass SparseZoo stub (ending in `base_none`, specifying the dense version) which pulls a dense BERT model trained on SST2 from the SparseZoo.

The SparseML script downloads the model, teacher, and dataset and recipe. Recipes are critical to the SparseML system, as they declare the algorithms and hyperparameters applied by SparseML. In this case, the premade transfer learning recipe for SST2 controls the number of epochs, the learning rate, as well as instructions for how to apply pruning, quantization, and distillation. SparseML parses the recipe and updates the training loop with the specified algorithms and hyperparameters encoded therein before starting training.

Here is what the recipe looks like:

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
   
The key `Modifiers` for sparse transfer learning are the following:
- `ConstantPruningModifier` instructs SparseML to maintain the sparsity structure of the network during the fine-tuning process
- `QuantizationModifier` instructs SparseML to apply quantization aware training to quantize the weights over the final epochs
- `DistillationModifier` instructs SparseML to apply model distillation at the logit layer
   
As a result, we end up with a 95% pruned and quantized BERT trained on SST2! It achieves ~92% accuracy.

#### Aside: Dense Teacher Creation (OPTIONAL)

The SparseML training scripts allow us to pass a dense teacher to apply model distillation during the training process. This is an optional 
parameter, but passing a dense teacher can help improve accuracy.

In the example above, we used SparseZoo stubs to specify and download dense teachers from SparseZoo. If you already have a Transformers-compatible model, you can use this as the dense teacher in place of training one from scratch by passing a local path to you model and configuration files. Alternatively, you can use the SparseML training script.

Here is the command we used to create the dense teacher hosted in SparseZoo:

```bash
sparseml.transformers.text_classification \
  --output_dir dense_obert-text_classification_sst2 \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/base-none \
  --recipe zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/base-none \
  --task_name sst2 --max_seq_length 128 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --preprocessing_num_workers 6 \
  --do_train --do_eval --evaluation_strategy epoch --fp16 --seed 20811 \
  --save_strategy epoch --save_total_limit 1
```

Note that the SparseZoo stubs passed to `--recipe` and `--model_name_or_path` here both end in `base-none`. This identifies the standard dense versions of the models/recipes in [SparseZoo](zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/base-none).

Here is what the `recipe` looks like:

```yaml
version: 1.1.0

# General Variables
num_epochs: 2
init_lr: 2e-5 
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

As you can see, this recipe only has modifiers for the learning rate schedule and number of epochs. As such, during the training
process, the model remains dense SparseML is not instructed to apply any pruning or quantization algorithms.

The dense teacher achieves ~92.9% validation accuracy.

### Exporting to ONNX

The SparseML installation provides a `sparseml.transformers.export_onnx` command that you can use to export the model to ONNX. Be sure the `--model_path` argument points to your trained model:

```bash
sparseml.transformers.export_onnx \
    --model_path ./pruned_quantized_obert-text_classification_sst2 \
    --task text_classification
```

The command creates a `./deployment` folder in your local directory, which contains the ONNX file and necessary Hugging Face tokenizer and configuration files.

## Other Dataset Examples

Let's walk through commands for other use cases. Here is an overview of some datasets we have transfered to:

| Use Case                   | Dataset                                                                       | Description                                                                                                                                                                                                          | Sparse Transfer Results  |
|----------------------------|-------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|
| Question Answering          | [SQuAD](https://huggingface.co/datasets/squad)                     | A reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text from the corresponding reading passage.  | 88.0 F1 (85.55 baseline)  |
| Binary Classification       | [QQP](https://huggingface.co/datasets/glue/viewer/qqp/train)       | A dataset made up of potential question pairs from Quora with a boolean label representing whether or not the questions are duplicates. | 91.08 acc (91.06 baseline)   |
| Multi-Class Classification  | [MultiNLI](https://huggingface.co/datasets/glue/viewer/mnli/train) | A crowd-sourced collection of sentence pairs annotated with textual entailment information. It covers a range of genres of spoken and written text and supports a distinctive cross-genre generalization evaluation. | 82.56 acc (84.53 baseline) |
| Multi-Label Classification  | [GoEmotions](https://huggingface.co/datasets/go_emotions)          | A dataset of Reddit comments labeled for 27 emotion categories or Neural (some comments have multiple).   | 48.82 avgF1 (49.85 baseline) |
| Sentiment Analysis          | [SST2](https://huggingface.co/datasets/conll2003)                  | A corpus that includes fine-grained sentiment labels for phrases within sentences and presents new challenges for sentiment compositionality.    | 91.97 acc (92.89 baseline) |
| Document Classification     | [IMDB](https://huggingface.co/datasets/imdb)                       | A large movie review dataset for binary sentiment analysis. Input sequences are long (vs SST2) | 93.16 acc (94.19 baseline) |
| Token Classification (NER)  | [CoNNL-2003](https://www.clips.uantwerpen.be/conll2003/ner/)       | A dataset concentrated on four types of named entities: persons, locations, organizations, and names of miscellaneous entities that do not belong to the previous three groups. | 98.55 acc (98.98 baseline) |

## Transfer Learning the Model

The following commands were used to generate the models:

- Question Answering (SQuAD)
```bash
sparseml.transformers.train.question_answering \
  --output_dir obert_base_pruned90_quant_squad \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --recipe zoo:nlp/question_answering/obert-base/pytorch/huggingface/squad/pruned90_quant-none \
  --distill_teacher zoo:nlp/question_answering/obert-base/pytorch/huggingface/squad/base-none \
  --dataset_name squad \
  --do_train --do_eval --evaluation_strategy epoch --logging_steps 1000 --save_steps 1000 \
  --per_device_train_batch_size 8 --per_device_eval_batch_size 32 --gradient_accumulation_steps 2 \
  --max_seq_length 384 --doc_stride 128 --preprocessing_num_workers 32 \
  --seed 42
```

- Text Classification: Binary Classification (QQP)
```bash
sparseml.transformers.train.text_classification \
  --output_dir obert_base_pruned90_quant_qqp \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --recipe zoo:nlp/text_classification/obert-base/pytorch/huggingface/qqp/pruned90_quant-none \
  --distill_teacher zoo:nlp/text_classification/obert-base/pytorch/huggingface/qqp/base-none \
  --task_name qqp \
  --do_train --do_eval --evaluation_strategy epoch --logging_steps 1000 \
  --save_steps 1000 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 \
  --max_seq_length 128 --preprocessing_num_workers 32 \
  --seed 10194
```

- Text Classification: Multi-Class Classification (MNLI)
```bash
sparseml.transformers.train.text_classification \
  --output_dir obert_base_pruned80_quant_mnli \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --recipe zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none \
  --distill_teacher zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/base-none \
  --task_name mnli \
  --do_train --do_eval --evaluation_strategy epoch --logging_steps 1000 --save_steps 1000 \
  --per_device_train_batch_size 8 --per_device_eval_batch_size 32 --gradient_accumulation_steps 4 \
  --max_seq_length 128 --preprocessing_num_workers 32 \
  --seed 5114
```

- Text Classification: Multi-Label Classification (GoEmotions)
```bash
sparseml.transformers.train.text_classification \
  --output_dir pruned_bert-multilabel_classification-goemotions \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --distill_teacher zoo:nlp/multilabel_text_classification/obert-base/pytorch/huggingface/goemotions/base-none \
  --recipe zoo:nlp/multilabel_text_classification/obert-base/pytorch/huggingface/goemotions/pruned90_quant-none \
  --dataset_name go_emotions --label_column_name labels --input_column_names text \
  --do_train --do_eval --fp16 --evaluation_strategy steps --eval_steps 200 \
  --logging_steps 200 --logging_first_step --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 --preprocessing_num_workers 8 \
  --max_seq_length 30 --save_strategy epoch --save_total_limit 1 \
  --seed 5550 \
```

- Text Classification: Document Classification (IMBD)
```bash
sparseml.transformers.train.text_classification \
  --output_dir obert-document_classification-imdb \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --distill_teacher zoo:nlp/document_classification/obert-base/pytorch/huggingface/imdb/base-none \
  --recipe zoo:nlp/document_classification/obert-base/pytorch/huggingface/imdb/pruned90_quant-none \
  --dataset_name imdb \
  --do_train --do_eval --validation_ratio 0.1 --fp16 \
  --evaluation_strategy steps --eval_steps 100 --logging_steps 100 --logging_first_step \
  --per_device_train_batch_size 8 --per_device_eval_batch_size 32 --gradient_accumulation_steps 5 \
  --save_strategy steps --save_steps 100 --save_total_limit 1
  --preprocessing_num_workers 6 --max_seq_length 512 \
  --seed 31222 \
```

- Text Classification: Sentiment Analysis (SST2)
```bash
sparseml.transformers.train.text_classification \
  --output_dir sparse_quantized_bert-text_classification_sst2 \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --recipe zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none \
  --distill_teacher zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/base-none \
  --task_name sst2 \
  --max_seq_length 128 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --preprocessing_num_workers 6 \
  --do_train --do_eval --evaluation_strategy epoch --fp16  \
  --save_strategy epoch --save_total_limit 1
```

- Token Classifcation: NER (Conll2003)
```bash
sparseml.transformers.train.token_classification \
  --output_dir sparse_bert-token_classification_connl2003 \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --distill_teacher zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/base-none \
  --recipe zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/pruned90_quant-none \
  --dataset_name conll2003 \
  --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --preprocessing_num_workers 6 \
  --do_train --do_eval --evaluation_strategy epoch --fp16 --seed 29204  \
  --save_strategy epoch --save_total_limit 1 
```

Check out the use case guides for more details on each task, including using a custom dataset and task specific arguments:
- [Sentiment Analysis](sentiment-analysis/sentiment-analysis-cli.md)
- [Text Classification](text-classification/text-classification-cli.md)
- [Token Classification](token-classification/token-classification-cli.md)
- [Question Answering](question-answering/question-answering-cli.md)

## Wrap-Up

Sparse transfer learning is just like typical fine-tuning - making it easy to create an inference-optimized 
NLP model trained on your dataset!

Checkout [DeepSparse](https://github.com/neuralmagic/deepsparse) for more details on reaching GPU-class performance by
deploying the spars emodels on CPUs!
