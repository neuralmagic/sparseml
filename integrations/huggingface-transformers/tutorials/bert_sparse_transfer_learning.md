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

# Sparse Transfer Learning With BERT

This tutorial shows how Neural Magic sparse models simplify the sparsification process by offering pre-sparsified BERT models for transfer learning onto other datasets.

## Overview

Neural Magic’s ML team creates sparsified models that allow anyone to plug in their data and leverage pre-sparsified models from the SparseZoo on top of Hugging Face's robust training pipelines.
Sparsifying involves removing redundant information from neural networks using algorithms such as pruning and quantization, among others.
This sparsification process results in many benefits for deployment environments, including faster inference and smaller file sizes.
Unfortunately, many have not realized the benefits due to the complicated process and number of hyperparameters involved.
Working through this tutorial, you will experience how Neural Magic recipes simplify the sparsification process by:

- Selecting a pre-sparsified model.
- Selecting a dataset use case.
- Creating a dense teacher model for distillation.
- Applying a sparse transfer learning recipe.
- Exporting an inference graph to reduce its deployment size and run it in an inference engine such as DeepSparse.

Before diving in, be sure to set up as listed in the [README](https://github.com/neuralmagic/sparseml/blob/main/integrations/huggingface-transformers/README.md) for this integration.
Additionally, all commands are intended to be run from the root of the `huggingface-transformers` integration folder (`cd integrations/huggingface-transformers`).

## Need Help?

For Neural Magic Support, sign up or log in to our [**Deep Sparse Community Slack**](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue.](https://github.com/neuralmagic/sparseml/issues)

## Selecting a Pre-sparsified Model

For this tutorial, you will use a 12-layer BERT model sparsified to 80% on the Wikitext and BookCorpus datasets. 
As a good tradeoff between inference performance and accuracy, 
this 12-layer model gives 3.9x better throughput while recovering close to the dense baseline for most transfer tasks.

The SparseZoo stub for this model is `zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/bookcorpus_wikitext/12layer_pruned80-none` and will be used to select the model in the training commands used later. 
Additional BERT models, including ones with higher sparsity and fewer layers, are found on the [SparseZoo](https://sparsezoo.neuralmagic.com/?page=1&domain=nlp&sub_domain=masked_language_modeling) and can be substituted in place of the 12-layer 80% sparse model for better performance or recovery.

## Selecting a Dataset Use Case

Results and examples are given for five common use cases with popular datasets in this tutorial. 
To apply these approaches to your own dataset, Hugging Face has additional information for setup of custom datasets [here](https://huggingface.co/transformers/custom_datasets.html).
Once you have successfully converted your dataset into Hugging Face's format, it can be safely plugged into these flows and used for sparse transfer learning from the pre-sparsified models.

The use cases and datasets covered in this tutorial are listed below along with results for sparse transfer learning with the 80% pruned BERT model.

| Use Case                   | Dataset                                                                       | Description                                                                                                                                                                                                          | Sparse Transfer Results  |
|----------------------------|-------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|
| Question and Answering     | [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)                          | A reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text from the corresponding reading passage.       | 87.1 F1 (87.5 baseline)  |
| Binary Classification      | [QQP](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs) | A dataset made up of potential question pairs from Quora with a boolean label representing whether or not the questions are duplicates                                                                              | 90 acc (91.3 baseline)   |
| Multi-Class Classification | [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/)                               | A crowd-sourced collection of sentence pairs annotated with textual entailment information. It covers a range of genres of spoken and written text and supports a distinctive cross-genre generalization evaluation. | 81.9 acc (82.3 baseline) |
| Named Entity Recognition   | [CoNNL-2003](https://www.clips.uantwerpen.be/conll2003/ner/)                   | A dataset concentrated on four types of named entities: persons, locations, organizations, and names of miscellaneous entities that do not belong to the previous three groups.                                       | 88.5 acc (94.7 baseline) |
| Sentiment Analysis         | [SST2](https://nlp.stanford.edu/sentiment/)                                   | A corpus that includes fine-grained sentiment labels for phrases within sentences and presents new challenges for sentiment compositionality.                                                                       | 91.3 acc (91.2 baseline) |

Once you have selected a use case and dataset, you are ready to create a teacher model.

## Creating a Dense Teacher

Distillation works very well for BERT and NLP in general to create highly sparse and accurate models for deployment.
Following this sentiment, you will create a dense teacher model before applying sparse transfer learning.
Note, the sparse models can be transferred without using distillation from the teacher; however, the end model's accuracy will be lower.
Additionally, if you have a dense model already, that can be used in place of the teacher to skip this step altogether.

The training commands for the teacher are listed below for each use case.
The batch size may need to be lowered depending on the available GPU memory. 
If you run out of memory or experience an initial crash, try to lower the batch size to remedy the issue.

| Use Case                   | Dataset                                                                       | Training Command                                                                                                                                                                                                                                                                                                            |
|----------------------------|-------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Question and Answering     | [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)                          | `sparseml.transformers.question_answering --model_name_or_path bert-base-uncased --dataset_name squad --do_train --do_eval --evaluation_strategy epoch --per_device_train_batch_size 16 --learning_rate 5e-5 --max_seq_length 384 --doc_stride 128 --output_dir models/teacher --num_train_epochs 2 --seed 2021`            |
| Binary Classification      | [QQP](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs) | `sparseml.transformers.text_classification --model_name_or_path bert-base-uncased --task_name qqp --do_train --do_eval --evaluation_strategy epoch --per_device_train_batch_size 32 --learning_rate 5e-5 --max_seq_length 128 --output_dir models/teacher --num_train_epochs 2 --seed 2021`                                 |
| Multi-Class Classification | [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/)                               | `sparseml.transformers.text_classification --model_name_or_path bert-base-uncased --task_name mnli --do_train --do_eval --evaluation_strategy epoch --per_device_train_batch_size 32 --learning_rate 5e-5 --max_seq_length 128 --output_dir models/teacher --num_train_epochs 2 --seed 2021`                                |
| Named Entity Recognition   | [CoNNL-2003](https://www.clips.uantwerpen.be/conll2003/ner/)                   | `sparseml.transformers.token_classification --model_name_or_path bert-base-uncased --dataset_name conll2003 --do_train --do_eval --evaluation_strategy epoch --per_device_train_batch_size 32 --learning_rate 5e-5 --output_dir models/teacher --preprocessing_num_workers 16 --num_train_epochs 5 --seed 2021`             |
| Sentiment Analysis         | [SST2](https://nlp.stanford.edu/sentiment/)                                   | `sparseml.transformers.text_classification --model_name_or_path bert-base-uncased --task_name sst2 --do_train --do_eval --evaluation_strategy epoch --per_device_train_batch_size 32 --learning_rate 5e-5 --max_seq_length 128 --output_dir models/teacher --preprocessing_num_workers 16 --num_train_epochs 2 --seed 2021` |

Select the training command that applies to your use case and then run it in your training environment.
The time to execute the training commands will differ according to dataset and training environment, but generally, they should run to completion in less than 12 hours.
Once the command has completed, you will have a deployable sparse model located in `models/teacher`.

You are ready to transfer learn the model.

## Transfer Learning the Model

With the dense teacher now trained to convergence, you will begin the sparse transfer learning with distillation with a recipe.
The teacher will distill knowledge into the sparse architecture, therefore increasing its performance while ideally converging to the dense solution's accuracy.
The recipe encodes the hyperparameters necessary for transfer learning the sparse architecture.
Specifically, it ensures that the sparsity is preserved through the training process.
The available recipes for the sparse BERT model you are using are visible on the [SparseZoo](https://sparsezoo.neuralmagic.com/models/nlp%2Fmasked_language_modeling%2Fbert-base%2Fpytorch%2Fhuggingface%2Fbookcorpus_wikitext%2F12layer_pruned80-none) along with recipes for the other models.

The transfer training commands are listed below for each use case.
As with training the teacher, the batch size may need to be lowered depending on the available GPU memory.
If you run out of memory or experience an initial crash, try to lower the batch size to remedy the issue.

| Use Case                   | Dataset                                                                       | Transfer Training Command                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|----------------------------|-------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Question and Answering     | [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)                          | `sparseml.transformers.question_answering --model_name_or_path zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/bookcorpus_wikitext/12layer_pruned80-none --distill_teacher models/teacher --dataset_name squad --do_train --do_eval --evaluation_strategy epoch --per_device_train_batch_size 16 --learning_rate 5e-5 --max_seq_length 384 --doc_stride 128 --preprocessing_num_workers 16 --output_dir models/12layer_pruned80-none --fp16 --seed 27942 --num_train_epochs 5 --recipe zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/bookcorpus_wikitext/12layer_pruned80-none?recipe_type=transfer-SQuAD --save_strategy epoch --save_total_limit 2`                          |
| Binary Classification      | [QQP](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs) | `sparseml.transformers.text_classification --model_name_or_path zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/bookcorpus_wikitext/12layer_pruned80-none --distill_teacher models/teacher --task_name qqp --do_train --do_eval --evaluation_strategy epoch --per_device_train_batch_size 32 --learning_rate 5e-5 --warmup_steps 11000 --output_dir models/12layer_pruned80-none --fp16 --seed 11712 --num_train_epochs 5 --recipe zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/bookcorpus_wikitext/12layer_pruned80-none?recipe_type=transfer-QQP --save_strategy epoch --save_total_limit 2`                                                                                |
| Multi-Class Classification | [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/)                               | `sparseml.transformers.text_classification --model_name_or_path zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/bookcorpus_wikitext/12layer_pruned80-none --distill_teacher models/teacher --task_name mnli --do_train --do_eval --evaluation_strategy epoch --per_device_train_batch_size 32 --learning_rate 5e-5 --warmup_steps 12000 --output_dir models/12layer_pruned80-none --fp16 --seed 27942 --num_train_epochs 5 --recipe zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/bookcorpus_wikitext/12layer_pruned80-none?recipe_type=transfer-MNLI --save_strategy epoch --save_total_limit 2`                                                                              |
| Named Entity Recognition   | [CoNNL-2003](https://www.clips.uantwerpen.be/conll2003/ner/)                   | `sparseml.transformers.token_classification --model_name_or_path zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/bookcorpus_wikitext/12layer_pruned80-none --distill_teacher models/teacher --dataset_name conll2003 --do_train --do_eval --evaluation_strategy epoch --per_device_train_batch_size 32 --learning_rate 5e-5 --preprocessing_num_workers 16 --output_dir models/12layer_pruned80-none --fp16 --seed 21097 --num_train_epochs 5 --recipe zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/bookcorpus_wikitext/12layer_pruned80-none?recipe_type=transfer-CoNLL2003 --save_strategy epoch --save_total_limit 2`                                                      |
| Sentiment Analysis         | [SST2](https://nlp.stanford.edu/sentiment/)                                   | `sparseml.transformers.text_classification --model_name_or_path zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/bookcorpus_wikitext/12layer_pruned80-none --distill_teacher models/teacher --task_name sst2 --do_train --do_eval --evaluation_strategy epoch --per_device_train_batch_size 32 --learning_rate 5e-5 --warmup_steps 2000 --max_seq_length 128 --preprocessing_num_workers 16 --output_dir models/12layer_pruned80-none --fp16 --seed 5922 --num_train_epochs 5 --recipe zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/bookcorpus_wikitext/12layer_pruned80-none?recipe_type=transfer-SST2 --save_strategy epoch --save_total_limit 2` |

Select the transfer training command that applies to your use case and then run it in your training environment.
The time to execute the training commands will differ according to dataset and training environment, but generally, they should run to completion in less than 12 hours.
Once the command has completed, you will have a sparse checkpoint located in `models/12layer_pruned80-none`.

You are ready to export for inference.

## Exporting for Inference

The `sparseml.transformers.export_onnx` scripts creates an ONNX model that can be used for deployment.
Running the `export_onnx` script with the `--help` command provides a full list of options.

The export commands are listed below for each use case and reference the output from the previously run transfer learning commands.

| Use Case                   | Dataset                                                                       | ONNX Export Command                                                                                                                             |
|----------------------------|-------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| Question and Answering     | [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)                          | `sparseml.transformers.export_onnx --task question-answering --model_path models/12layer_pruned80-none                                          |
| Binary Classification      | [QQP](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs) | `sparseml.transformers.export_onnx --task text-classification --finetuning_task qqp --model_path models/12layer_pruned80-none                   |
| Multi-Class Classification | [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/)                               | `sparseml.transformers.export_onnx --task text-classification --finetuning_task mnli --model_path models/12layer_pruned80-none              |
| Named Entity Recognition   | [CoNNL-2003](https://www.clips.uantwerpen.be/conll2003/ner/)                   | `sparseml.transformers.export_onnx --task token-classification --finetuning_task conll2003 --model_path models/12layer_pruned80-none           |
| Sentiment Analysis         | [SST2](https://nlp.stanford.edu/sentiment/)                                   | `sparseml.transformers.export_onnx --task text-classification --finetuning_task sst2 --model_path models/12layer_pruned80-none --task_name sst2 |

Select the export command that applies to your use case and then run in your training environment.
Once the command has completed, you will have a deployable sparse model located in the given `model_directory`.

Now you can run the `.onnx` file through a compression algorithm to reduce its deployment size and run it in ONNX-compatible inference engines such as [DeepSparse](https://github.com/neuralmagic/deepsparse).
The DeepSparse Engine is explicitly coded to support running sparsified models for significant improvements in inference performance.
An example for benchmarking and deploying BERT models with DeepSparse can be found [here](https://github.com/neuralmagic/deepsparse/tree/main/examples/huggingface-transformers).
When running in `DeepSparse`, reference the entire model directory so that the ONNX file as well as tokenizer and data configs
are read.

## Wrap-Up

Neural Magic sparse models and recipes simplify the sparsification process by enabling sparse transfer learning to create highly accurate pruned BERT models.
In this tutorial, you selected a pre-sparsified model, applied a Neural Magic recipe for sparse transfer learning, and exported it to ONNX to run through an inference engine.

Now, refer [here](https://github.com/neuralmagic/deepsparse/blob/main/src/deepsparse/transformers/README.md) for an example for benchmarking and deploying transformers models with DeepSparse.

For Neural Magic Support, sign up or log in to our [**Deep Sparse Community Slack**](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue.](https://github.com/neuralmagic/sparseml/issues)
