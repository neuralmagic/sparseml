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

# **SparseML CLI: Sparse Transfer Learning with BERT**

This page explains how to fine-tune a pre-sparsified BERT model onto a downstream dataset with SparseML's CLI.

## **Sparse Transfer Learning Overview**

Sparse Transfer Learning is quite similiar to typical NLP transfer learning, where we fine-tune a checkpoint pretrained on a large dataset like WikipediaBookCorpus onto a smaller downstream dataset and task. However, with Sparse Transfer Learning, we simply start the fine-tuning process from a pre-sparsified model and maintain sparsity while the training process occurs.

SparseZoo contains pre-sparsified checkpoints of common NLP models like BERT and RoBERTa. These models can be used as the starting checkpoint for the sparse transfer learning workflow.

[Check out the full list of pre-sparsified models](https://sparsezoo.neuralmagic.com/?domain=nlp&sub_domain=masked_language_modeling&page=1)

## **Installation**

Install via `pip`:

```
pip install sparseml[transformers]
```

## **Example: Sparse Transfer Learning onto SST2**

Let's try a simple example of fine-tuning a pre-sparsified model onto the SST dataset. SST2 is a sentiment analysis
dataset, with each sentence labeled with a 0 or 1 representing negative or positive sentiment.

[SST2 Dataset Card](https://huggingface.co/datasets/glue/viewer/sst2/train)

### **Selecting a Pre-Sparsified Model**

We will fine-tune a [90% pruned BERT-base](https://sparsezoo.neuralmagic.com/models/nlp%2Fmasked_language_modeling%2Fobert-base%2Fpytorch%2Fhuggingface%2Fwikipedia_bookcorpus%2Fpruned90-none), identified by the following stub:
```bash
zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none
```

### **Kick off Training**

We will use SparseML's `sparseml.transformers.text_classification` training script. 

To run sparse transfer learning, we first need to create/select a sparsification recipe. For sparse transfer, we need a recipe that instructs SparseML to maintain sparsity during training and to quantize the model over the final epochs. 

For the SST2 dataset, there is a [transfer learning recipe available in SparseZoo](https://sparsezoo.neuralmagic.com/models/nlp%2Fsentiment_analysis%2Fobert-base%2Fpytorch%2Fhuggingface%2Fsst2%2Fpruned90_quant-none), identified by the following stub:
```
zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none
```

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

Run the following to fine-tune the 90% pruned BERT model on the SST2 dataset:

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

Let's discuss the key arguments:
- `--task sst2` specifies the dataset to train on (in this case SST2). You can pass any GLUE task to the `--task` command. Check out the use case pages for passing a custom dataset.

- `--model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none` specifies the starting checkpoint for the training process. Here, we passed a SparseZoo stub, which identifies the 90% pruned BERT model in the SparseZoo. The script downloads the PyTorch model to begin training. In addition to SparseZoo stubs, you can also pass a local path to a PyTorch checkpoint.

- `--recipe zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none` specifies the transfer learning recipe. In this case, we passed a SparseZoo stub, which instructs SparseML to download a premade SST2 transfer learning recipe. In addition to SparseZoo stubs, you can also pass a local path to a YAML recipe.

- `--distill_teacher zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/base-none` is an optional argument that allows you to apply model distillation during fine-tuning. Here, we pass SparseZoo stub (ending in base_none, specifying the dense version) which pulls a dense BERT model trained on SST2 from the SparseZoo. In addition to SparseZoo stubs, you can also pass a local path to a PyTorch checkpoint.

The script uses the SparseZoo stubs to identify and download the starting checkpoint and recipe file. SparseML then parses the transfer learning recipe and adjusts the training loop to maintain sparsity during the fine-tuning process. It then kicks off the transfer learning run.

The resulting model is 90% pruned and quantized, and achieves 92% validation accuracy on SST2!

### **Aside: Dense Teacher Creation**

You will notice that we passed a `--distill_teacher` argument to the training loop above. This is an optional argument, but distillation can help to improve accuracy during the transfer learning process.

In the example above, we used a SparseZoo stub to download a teacher model from the Zoo. However, you can also train your own teacher model. While you are free to train the teacher in whatever manner you
want, you can also use the SparseML training script.

Run the following to fine-tune a dense BERT model from the SparseZoo on the SST2 dataset:

```bash
sparseml.transformers.text_classification \
  --output_dir dense_obert-text_classification_sst2 \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/base-none \
  --recipe zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/base-none \
  --task_name sst2 --max_seq_length 128 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --preprocessing_num_workers 6 \
  --do_train --do_eval --evaluation_strategy epoch --fp16 --seed 20811 \
  --save_strategy epoch --save_total_limit 1
```

The resulting model achieves 92.9% validation accuracy.

To use the locally trained teacher, update the sparse transfer command to use `--distill_teacher ./dense_obert-text_classification_sst2`.

Note that we still passed `--recipe zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/base-none` during dense training. The SparseZoo stub ends in `base-none`, which identifies a transfer learning recipe that was used to train the dense model. 

Here's what the recipe for the dense model looks like:

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

You will notice the modifiers only specify the learning rate schedule and number of epochs. As such, SparseML does not apply any sparsity related algorithms, so the training occurs as usual.

### **Export to ONNX**

SparseML provides a `sparseml.transformers.export_onnx` command that you can use to export the model to ONNX. Be sure the `--model_path` argument points to your trained model:

```bash
sparseml.transformers.export_onnx \
 --model_path ./pruned_quantized_obert-text_classification_sst2 \
 --task text_classification
```

The command creates a `./deployment` folder in your local directory, which contains the ONNX file and necessary Hugging Face tokenizer and configuration files for deployment with DeepSparse.

## **Other Dataset Examples**

Let's walk through commands for other use cases. Here is an overview of some datasets we have transfered to:

| Use Case                   | Dataset                                                                       | Description                                                                                                                                                                                                          | Sparse Transfer Results  |
|----------------------------|-------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|
| Question Answering          | [SQuAD](https://huggingface.co/datasets/squad)                     | A reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text from the corresponding reading passage.  | 88.0 F1 (85.55 baseline)  |
| Binary Classification       | [QQP](https://huggingface.co/datasets/glue/viewer/qqp/train)       | A dataset made up of potential question pairs from Quora with a boolean label representing whether or not the questions are duplicates. | 91.08 acc (91.06 baseline)   |
| Multi-Class Classification  | [MultiNLI](https://huggingface.co/datasets/glue/viewer/mnli/train) | A crowd-sourced collection of sentence pairs annotated with textual entailment information. It covers a range of genres of spoken and written text and supports a distinctive cross-genre generalization evaluation. | 82.56 acc (84.53 baseline) |
| Multi-Label Classification  | [GoEmotions](https://huggingface.co/datasets/go_emotions)          | A dataset of Reddit comments labeled for 27 emotion categories or Neutral (some comments have multiple).   | 48.82 avgF1 (49.85 baseline) |
| Sentiment Analysis          | [SST2](https://huggingface.co/datasets/conll2003)                  | A corpus that includes fine-grained sentiment labels for phrases within sentences and presents new challenges for sentiment compositionality.    | 91.97 acc (92.89 baseline) |
| Document Classification     | [IMDB](https://huggingface.co/datasets/imdb)                       | A large movie review dataset for binary sentiment analysis. Input sequences are long (vs SST2) | 93.16 acc (94.19 baseline) |
| Token Classification (NER)  | [CoNNL-2003](https://www.clips.uantwerpen.be/conll2003/ner/)       | A dataset concentrated on four types of named entities: persons, locations, organizations, and names of miscellaneous entities that do not belong to the previous three groups. | 98.55 acc (98.98 baseline) |

## **Transfer Learning**

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
  --output_dir obert_base_pruned90_quant_mnli \
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
  --seed 5550
```

- Text Classification: Document Classification (IMDB)
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
  --save_strategy steps --save_steps 100 --save_total_limit 1 \
  --preprocessing_num_workers 6 --max_seq_length 512 \
  --seed 31222
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

## **Wrap-Up**

Check out the use case guides for more details on using a custom dataset, using a custom trainer, and task-specific arguments:
- [Sentiment Analysis](sentiment-analysis/sentiment-analysis-cli.md)
- [Text Classification](text-classification/text-classification-cli.md)
- [Token Classification](token-classification/token-classification-cli.md)
- [Question Answering](question-answering/question-answering-cli.md)
