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

# SparseML Hugging Face Integration

By integrating with robust training flows in the Transformers repository, SparseML enables you to train inference-optimized sparse versions of NLP models like BERT on your dataset.

There are two pathways:
- **Sparse Transfer Learning** - fine-tune a pre-sparsified NLP checkpoint on your own dataset **[RECOMMENDED]**
- **Sparsification from Scratch** - apply pruning and quantization to sparsify any of the `transformer` models from scratch.

Once trained, SparseML enables you to export models to the ONNX format, such that they can be deployed with DeepSparse for GPU-class performance on the CPU.

## Installation

Install with `pip`:

```bash
pip install sparseml[torch]
```

**Note**: Transformers will not immediately install with this command. Instead, a sparsification-compatible version of Transformers will install on the first invocation of the Transformers code in SparseML.

## Tutorials

- [Sparse Transfer Learning](sparse-transfer-learning-bert.md) [**RECOMMENDED**]
- Sparsification from Scratch (example coming soon!)

### Use Case Examples - CLI
- [Sparse Transfer Learning for Sentiment Analysis](tutorials/sentiment-analysis/sentiment-analysis-cli.md)
- [Sparse Transfer Learning for Text Classification](tutorials/text-classification/text-classification-cli.md)
- [Sparse Transfer Learning for Token Classification](tutorials/token-classification/token-classification-cli.md)
- Sparse Transfer Learning for Question Answering (example coming soon!)
- Sparsifying from Scratch (example coming soon!)

### Use Case Examples - Python

#### Sentiment Analysis (Single Input Binary Text Classification)
- [Sparse Transfer with GLUE Datasets (SST2) for sentiment analysis](tutorials/sentiment-analysis/docs-sentiment-analysis-python-sst2.ipynb)
- [Sparse Transfer with Custom Datasets (RottenTomatoes) and Custom Teacher from HF Hub for sentiment analysis](tutorials/sentiment-analysis/docs-sentiment-analysis-python-custom-teacher-rottentomatoes)

#### Text Classification (Single / Multi Input Text Classification)
- [Sparse Transfer with GLUE Datasets (QQP) for multi-input text classification](tutorials/text-classification/docs-text-classification-python-qqp.ipynb)
- [Sparse Transfer with Custom Datasets (SICK) for multi-input text classification](tutorials/text-classification/docs-text-classification-python-sick.ipynb)
- [Sparse Transfer with Custom Datasets (TweetEval) and Custom Teacher for single input text classificaiton](tutorials/text-classification/docs-text-classification-python-custom-teacher-tweeteval.ipynb)
- [Sparse Transfer with Custom Datasets (GoEmotions) for multi-label text classification](tutorials/text-classification/docs-text-classification-python-multi-label-go_emotions.ipynb)

#### Token Classification 
- [Sparse Transfer with Conll2003 for named-entity-recognition](tutorials/token-classification/docs-token-classification-conll2003.ipynb)
- [Sparse Transfer with Custom Datasets (WNUT) and Custom Teacher for named-entity-recognition](tutorials/token-classification/docs-token-classification-custom-teacher-wnut.ipynb)

#### Question Answering

- Sparse Transfer with SQuAD (Example coming soon!)

#### General 

- Sparsifying from Scratch (Example coming soon!)

## Quick Tour

### SparseZoo

Neural Magic has pre-sparsified many common models, including BERT-base, BERT-large, DistillBERT, and RoBERTa. These models and associated sparsification recipes can be deployed directly or can be fine-tuned onto custom dataset via sparse transfer learning. This makes it easy to create a sparse version of model trained on your dataset.

Check out the model cards in the [SparseZoo](https://sparsezoo.neuralmagic.com/?repo=huggingface&page=1).

### Recipes

SparseML Recipes are YAML files that encode the instructions for sparsifying a model or sparse transfer learning. The SparseML CLI and Python API accept the recipes as inputs, parse the instructions, and apply the specified algorithms and hyperparameters during the training process.

### SparseML CLI

SparseML's CLI enables you to kick-off sparsification workflows with various utilities like creating training pipelines, dataset loading, checkpoint saving, metric reporting, and logging handled for you. Appending the `--help` argument will provide a full list of options for training in SparseML:

```bash
sparseml.transformers.[task] --help
```

output:
```bash
--output_dir:                  The directory in which to store the outputs from the training runs such as results, the trained model, and supporting files.
--model_name_or_path:          The path or SparseZoo stub for the model to load for training.
--recipe:                      The path or SparseZoo stub for the recipe to use to apply sparsification algorithms or sparse transfer learning to the model.
--distill_teacher:             The path or SparseZoo stub for the teacher to load for distillation.
--dataset_name or --task_name: The dataset or task to load for training.
```

Currently supported tasks include: 
- `masked_language_modeling`
- `text_classification`
- `token_classification`
- `question_answering`

### SparseML Python API

For additional flexibility, SparseML also offers a `Trainer` class that inherits from the familiar [Transformers's Trainer](https://huggingface.co/docs/transformers/main_classes/trainer). 

SparseML's `Trainer` inherits all of the functionality from the Transformers repository, but also accepts a `recipe`, which allows you to specify sparsity-related algorithms and hyperparameters. The `Trainer` parses the recipe and adjusts the training loop to apply sparsification algorithms or sparse transfer learning. As such, you can leverage Hugging Face's friendly utilities such as the Model Hub, `AutoTokenizers`, `AutoModels`, and `datasets` in concert with SparseML's sparsity-related algorithms!

We create the `Trainer` and kick of a run like this:

```python
from sparseml.transformers.sparsification import Trainer, TrainingArguments

def run_training(model, model_path, recipe_path, teacher, training_args, dataset, tokenizer, compute_metrics):
    # setup training loop based on recipe passed
    trainer = Trainer(
        recipe=recipe_path,
        model=model,
        model_state_path=model_path,
        distill_teacher=teacher,
        metadata_args=["per_device_train_batch_size","per_device_eval_batch_size","fp16"],
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # run training
    trainer.train(resume_from_checkpoint=False)
```

Check out the tutorials for actual working examples using the `Trainer` class.

## Quick Start: Sparse Transfer Learning

### Overview

Sparse Transfer is quite similiar to the typical transfer learing process used to train NLP models, where we fine-tune a pretrained checkpoint onto a smaller downstream dataset. With Sparse Transfer Learning, we simply start the fine-tuning process from a pre-sparsified checkpoint and maintain sparsity while the training process occurs.

In this example, we will fine-tune a [90% pruned version of BERT](https://sparsezoo.neuralmagic.com/models/nlp%2Fmasked_language_modeling%2Fobert-base%2Fpytorch%2Fhuggingface%2Fwikipedia_bookcorpus%2Fpruned90-none) onto SST2.

### Dense Teacher Creation

To support the transfer learning process, we can (optionally) apply model distillation. To enable distillation, we first create a dense teacher model. If you already have a Transformers-compatible model, you can use this as the dense teacher in place of training one from scratch.

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

### Kick off Training

We can start the Sparse Transfer Learning by passing a starting checkpoint  and recipe to the training script. For Sparse Transfer, we will use a recipe that instructs SparseML to maintain sparsity during training and to quantize the model. For the SST2 dataset, there is a [transfer learning recipe available in SparseZoo](https://sparsezoo.neuralmagic.com/models/nlp%2Fsentiment_analysis%2Fobert-base%2Fpytorch%2Fhuggingface%2Fsst2%2Fpruned90_quant-none), identified by the following SparseZoo stub:
```
zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none
```

<details>
   <summary>Click to see the recipe</summary>
</br>

SparseML parses the `Modifers` in the recipe and updates the training loop with logic encoded therein.
   
The key `Modifiers` for sparse transfer learning are the following:
- `ConstantPruningModifier` instructs SparseML to maintain the sparsity structure of the network during the fine-tuning process
- `QuantizationModifier` instructs SparseML to apply quantization aware training to quantize the weights over the final epochs
- `DistillationModifier` instructs SparseML to apply model distillation at the logit layer
   
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

Run the following to sparse transfer learn the 90% pruned BERT model on the SST2 dataset:
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

The script uses the SparseZoo stubs to identify and download the starting checkpoint and YAML-based recipe file from the SparseZoo. SparseML parses the transfer learning recipe and adjusts the trainign process to maintain sparsity during the fine-tuning process.

The resulting model is 90% pruned and quantized, and achieves 92% validation accuracy on SST2!

Keep in mind that the `--distill_teacher` argument is set to pull a dense SST2 model from the SparseZoo. If you trained a dense teacher with the command from above, update the script to use `--distill_teacher ./dense_obert-text_classification_sst2`.

### Export to ONNX

The SparseML installation provides a `sparseml.transformers.export_onnx` command that you can use to export the model to ONNX. Be sure the `--model_path` argument points to your trained model:

```bash
sparseml.transformers.export_onnx \
    --model_path ./pruned_quantized_obert-text_classification_sst2 \
    --task text_classification
```

The command creates a `./deployment` folder in your local directory, which contains the ONNX file and necessary Hugging Face tokenizer and configuration files.

### DeepSparse Deployment

Now that the model is in an ONNX format, it is ready for deployment with the DeepSparse. 

Run the following command to install it:

```bash
pip install deepsparse
```

#### DeepSparse Pipelines

The Python code below gives an example for using the DeepSparse Pipeline API with sentiment analysis. Be sure to change out the `model_path` argument for the model folder of your trained model:

```python
from deepsparse import Pipeline

# create pipeline, compile model
model_path = "./deployment"
sa_pipeline = Pipeline.create(task="sentiment-analysis", model_path=model_path)

# run inference with deepsparse making the predictions on the CPU!
inference = sa_pipeline("I love using DeepSparse to speed up my inferences")
print(inference)
# >> labels=['positive'] scores=[0.9972139000892639]
```

#### Other Deployment Options

Checkout the DeepSparse GitHub repository for more details.
