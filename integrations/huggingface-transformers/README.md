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

# **SparseML Hugging Face Integration**

This directory explains how to use SparseML's `transformers` integration to train inference-optimized sparse NLP models on your dataset.

There are two main workflows enabled by SparseML:
- **Sparse Transfer Learning** - fine-tune a pre-sparsified checkpoint on your own dataset **[RECOMMENDED]**
- **Sparsification from Scratch** - apply pruning and quantization to sparsify `transformer` models from scratch

Once trained, SparseML enables you to export models to the ONNX format, such that they can be deployed with DeepSparse.

## **Installation**

Install with `pip`:

```bash
pip install sparseml[transformers]
```

## **Tutorials**

- [Sparse Transfer Learning with the Python API](tutorials/sparse-transfer-learning-bert-python.md) [**RECOMMENDED**]
- [Sparse Transfer Learning with the CLI](tutorials/sparse-transfer-learning-bert.md) [**RECOMMENDED**]
- Sparsification from Scratch (example coming soon!)

### **Use Case Examples - CLI**
- [Sparse Transfer Learning for Sentiment Analysis](tutorials/sentiment-analysis/sentiment-analysis-cli.md)
- [Sparse Transfer Learning for Text Classification](tutorials/text-classification/text-classification-cli.md)
- [Sparse Transfer Learning for Token Classification](tutorials/token-classification/token-classification-cli.md)
- [Sparse Transfer Learning for Question Answering](tutorials/question-answering/question-answering-cli.md)
- Sparsifying from Scratch (example coming soon!)

### **Use Case Examples - Python**

- [Sparse Transfer with GLUE Datasets (SST2) for sentiment analysis](tutorials/sentiment-analysis/docs-sentiment-analysis-python-sst2.ipynb)
- [Sparse Transfer with Custom Datasets (RottenTomatoes) and Custom Teacher from HF Hub for sentiment analysis](tutorials/sentiment-analysis/docs-sentiment-analysis-python-custom-teacher-rottentomatoes.ipynb)
- [Sparse Transfer with GLUE Datasets (QQP) for multi-input text classification](tutorials/text-classification/docs-text-classification-python-qqp.ipynb)
- [Sparse Transfer with Custom Datasets (SICK) for multi-input text classification](tutorials/text-classification/docs-text-classification-python-sick.ipynb)
- [Sparse Transfer with Custom Datasets (TweetEval) and Custom Teacher for single input text classificaiton](tutorials/text-classification/docs-text-classification-python-custom-teacher-tweeteval.ipynb)
- [Sparse Transfer with Custom Datasets (GoEmotions) for multi-label text classification](tutorials/text-classification/docs-text-classification-python-multi-label-go_emotions.ipynb)
- [Sparse Transfer with Conll2003 for named entity recognition](tutorials/token-classification/docs-token-classification-python-conll2003.ipynb)
- [Sparse Transfer with Custom Datasets (WNUT) and Custom Teacher for named entity recognition](tutorials/token-classification/docs-token-classification-python-custom-teacher-wnut.ipynb)
- Sparse Transfer with SQuAD (example coming soon!)
- Sparse Transfer with Squadshifts Amazon (example coming soon!)

## **Quick Tour**

### **SparseZoo**

SparseZoo is an open-source repository of pre-sparsified models, including BERT-base, BERT-large, RoBERTa-base, RoBERTa-large, and DistillBERT. With SparseML, you can fine-tune these pre-sparsified checkpoints onto custom datasets (while maintaining sparsity) via sparse transfer learning. This makes training inference-optimized sparse models almost identical to your typical training workflows!

[Check out the available models](https://sparsezoo.neuralmagic.com/?repos=huggingface)

### **Recipes**

Recipes are YAML files that encode the instructions for sparsifying a model or sparse transfer learning. SparseML accepts the recipes as inputs, parses the instructions, and applies the specified algorithms and hyperparameters during the training process.

In such a way, recipes are the declarative interface for specifying which sparsity-related algorithms to apply!

### **SparseML Python API**

Because of the declarative, recipe-based approach, you can add SparseML to your `transformers` training pipelines via SparseML's `Trainer` class.

Inheriting from the familiar [Hugging Face `Trainer`](https://huggingface.co/docs/transformers/main_classes/trainer), SparseML's `Trainer` extends the functionality to enable passing a `recipe`. This allows you to specify the sparsity related algorithms and hyperparameters that should be applied in the training process. SparseML's `Trainer` parses the recipe and adjusts the training loop to apply the specified algorithms.

As such, you can swap the SparseML `Trainer` into your existing `transformers` training pipelines, leveraging Hugging Face's friendly utilities like Model Hub, `AutoTokenizers`, `AutoModels`, and `datasets` in concert with SparseML's sparsity-related algorithms!

The following demonstrates sample usage:

```python
from sparseml.transformers.sparsification import Trainer, TrainingArguments

def run_training(model, model_path, recipe_path, training_args, dataset, tokenizer, compute_metrics):
    # setup training loop based on recipe
    trainer = Trainer(
        recipe=recipe_path,
        model=model,
        model_state_path=model_path,
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

Note that the `model`, `training_args`, `dataset`, `tokenizer`, and `compute_metrics` function are all standard Hugging Face classes. We simply swap in the SparseML `Trainer` and `TrainingArguments` and pass a `recipe` and we are off and running!

Check out the tutorials for actual working examples using the `Trainer` class.

### **SparseML CLI**

In addition to the code-level API, SparseML offers pre-made training pipelines for common NLP tasks via the CLI interface.

The CLI enables you to kick-off training runs with various utilities like dataset loading and pre-processing, checkpoint saving, metric reporting, and logging handled for you. Appending the `--help` argument will provide a full list of options for training in SparseML:

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

## **Quick Start: Sparse Transfer Learning with the CLI**

### **Sparse Transfer Learning Overview**

Sparse Transfer is very similiar to the typical transfer learing process used to train NLP models, where we fine-tune a checkpoint pretrained on a large upstream dataset using masked language modeling onto a smaller downstream dataset. With Sparse Transfer Learning, however, we simply start the fine-tuning process from a pre-sparsified checkpoint and maintain sparsity while the training process occurs.

Here, we will fine-tune a [90% pruned version of BERT](https://sparsezoo.neuralmagic.com/models/obert-base-wikipedia_bookcorpus-pruned90?comparison=obert-base-wikipedia_bookcorpus-base) from the SparseZoo onto SST2.

### **Kick off Training**

We will use SparseML's `sparseml.transformers.text_classification` training script.

To run sparse transfer learning, we first need to create/select a sparsification recipe. For sparse transfer, we need a recipe that instructs SparseML to maintain sparsity during training and to quantize the model. 

For the SST2 dataset, there is a [transfer learning recipe available in SparseZoo](https://sparsezoo.neuralmagic.com/models/obert-base-sst2_wikipedia_bookcorpus-pruned90_quantized?comparison=obert-base-sst2_wikipedia_bookcorpus-base&tab=0), identified by the following SparseZoo stub:
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

- `--distill_teacher zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/base-none` is an optional argument that allows you to apply model distillation during fine-tuning. Here, we pass SparseZoo stub (ending in base_none, specifying the dense version) which pulls a dense BERT model trained on SST2 from the SparseZoo. In addition to SpareseZoo stubs, you can also pass a local path to a PyTorch checkpoint.

The script uses the SparseZoo stubs to identify and download the starting checkpoint and recipe file. SparseML then parses the transfer learning recipe and adjusts the training loop to maintain sparsity during the fine-tuning process. It then kicks off the transfer learning run.

The resulting model is 90% pruned and quantized, and achieves 92% validation accuracy on SST2!

### **Aside: Dense Teacher Creation**

You will notice that we passed a `--distill_teacher` argument to the training loop above. This is an optional argument, but distillation can help to improve accuracy during the transfer learning process. Above, we used a SparseZoo stub to download a teacher model from the Zoo. However, you can also train your own teacher model. While you are free to train the teacher in whatever manner you
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

To use the locally trained dense teacher, update the sparse transfer command to use `--distill_teacher ./dense_obert-text_classification_sst2`.

Note that we still passed `--recipe zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/base-none` during dense training. You will notice that
the SparseZoo stub ends in `base-none`. This identifies a transfer learning recipe that was used to train the dense model. 

You can see that this recipe contains only hyperparameters for the learing rate and number of epochs:

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

As such, SparseML does not apply any sparsity related algorithms, so the training occurs as usual.

### **Export to ONNX**

The SparseML installation provides a `sparseml.transformers.export_onnx` command that you can use to export the model to ONNX. Be sure the `--model_path` argument points to your trained model:

```bash
sparseml.transformers.export_onnx \
  --model_path ./pruned_quantized_obert-text_classification_sst2 \
  --task text_classification
```

The command creates a `./deployment` folder in your local directory, which contains the ONNX file and necessary Hugging Face tokenizer and configuration files.

### **DeepSparse Deployment**

Now that the model is in an ONNX format, it is ready for deployment with the DeepSparse. 

Run the following command to install it:

```bash
pip install deepsparse
```

#### **DeepSparse Pipelines**

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
