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

# **SparseML Python API: Sparse Transfer Learning with BERT**

This page explains how to fine-tune a pre-sparsified BERT model onto a downstream dataset with SparseML's `Trainer`.

## **Sparse Transfer Learning Overview**

Sparse Transfer Learning is quite similiar to typical NLP transfer learning, where we fine-tune a checkpoint pretrained on a large dataset like WikipediaBookCorpus onto a smaller downstream dataset and task. However, with Sparse Transfer Learning, we simply start the fine-tuning process from a pre-sparsified model and maintain sparsity while the training process occurs.

SparseZoo contains pre-sparsified checkpoints of common NLP models like BERT and RoBERTa. These models can be used as the starting checkpoint for the sparse transfer learning workflow.

[Check out the full list of pre-sparsified models](https://sparsezoo.neuralmagic.com/?domain=nlp&sub_domain=masked_language_modeling&page=1)

## **Installation**

Install via `pip`:

```
pip install sparseml[transformers]
```

## **Sparse Transfer Learning onto SST2**

Let's try a simple example of fine-tuning a pre-sparsified model onto the SST dataset. SST2 is a sentiment analysis
dataset, with each sentence labeled with a 0 or 1 representing negative or positive sentiment.

### **Step 1: Download Pre-Sparsified Checkpoint**

We will fine-tune a [90% pruned BERT-base](https://sparsezoo.neuralmagic.com/models/nlp%2Fmasked_language_modeling%2Fobert-base%2Fpytorch%2Fhuggingface%2Fwikipedia_bookcorpus%2Fpruned90-none), identified by the following stub:
```bash
zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none
```

Run the following to download it:
```python
from sparsezoo import Model
model_stub = "zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none" 
zoo_model = Model(model_stub, download_path="./model")
model_path = zoo_model.training.path 
```

Additionally, SparseML allows you to apply model distillation from a dense teacher model during the fine-tuning process. This is an optional step, but it can help increase accuracy.

For SST2, there is a [dense BERT-base](https://sparsezoo.neuralmagic.com/models/nlp%2Fsentiment_analysis%2Fobert-base%2Fpytorch%2Fhuggingface%2Fsst2%2Fbase-none) trained on SST2, identified by the following stub:

```bash
zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/base-none
```

Run the following to download it:
```python
from sparsezoo import Model
teacher_stub = "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/base-none"
zoo_model = Model(teacher_stub, download_path="./teacher")
teacher_path = zoo_model.training.path 
```

### **Step 2: Setup Hugging Face Model Objects**

With the models downloaded, we will set up the Hugging Face `tokenizer`, `config`, and `model`. These are all native Hugging Face objects, so check out the Hugging Face docs for more details on `AutoModel`, `AutoConfig`, and `AutoTokenizer` as needed. 

We instantiate these classes by passing the local path to the directory containing the `pytorch_model.bin`, `tokenizer.json`, and `config.json` files from the SparseZoo download.

```python
from sparseml.transformers.utils import SparseAutoModel
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer

NUM_LABELS = 2

# see examples for how to use models with different tokenizers
tokenizer = AutoTokenizer.from_pretrained(model_path)

# setup configs
model_config = AutoConfig.from_pretrained(model_path, num_labels=NUM_LABELS)
teacher_config = AutoConfig.from_pretrained(teacher_path, num_labels=NUM_LABELS)

# load model from local directory (model_path)
model_kwargs = {"config": model_config}
model_kwargs["state_dict"], s_delayed = SparseAutoModel._loadable_state_dict(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, **model_kwargs,)

# load model from local directory (teacher_path)
teacher_kwargs = {'config':teacher_config}
teacher_kwargs["state_dict"], t_delayed = SparseAutoModel._loadable_state_dict(teacher_path)
teacher = AutoModelForSequenceClassification.from_pretrained(teacher_path, **teacher_kwargs,)
```

### **Step 3: Prepare a Dataset**

Next, download a dataset and prepare it for training. We can use the Hugging Face [`datasets`](https://huggingface.co/docs/datasets/index) library.

Begin by loading the [SST2](https://huggingface.co/datasets/glue/viewer/sst2/train) dataset:

```python
from datasets import load_dataset
from pprint import pprint

# load dataset natively
dataset = load_dataset("glue", "sst2")
pprint(dataset["train"][100])

# >> {'idx': 0,
# >> 'label': 0,
# >> 'sentence': 'hide new secretions from the parental units '}
```

Next, tokenize the dataset using the `tokenizer` created above.

```python
def preprocess_fn(examples):
  return tokenizer(examples["sentence"], 
                   padding="max_length", 
                   max_length=min(tokenizer.model_max_length, 128), 
                   truncation=True)

tokenized_dataset = dataset.map(preprocess_fn, batched=True)
```

### **Step 4: Create a Recipe** 

To run sparse transfer learning, we first need to create/select a sparsification recipe. For sparse transfer, we need a recipe that instructs SparseML to maintain sparsity during training and to quantize the model over the final epochs.

For the SST2 dataset, there is a [transfer learning recipe available in SparseZoo](https://sparsezoo.neuralmagic.com/models/nlp%2Fsentiment_analysis%2Fobert-base%2Fpytorch%2Fhuggingface%2Fsst2%2Fpruned90_quant-none), identified by the following stub:
```
zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none
```

We will use this recipe for the example. This is what it looks like:
   
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

Run the following to download the recipe to your local directory:

```python
from sparsezoo import Model
transfer_stub = "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none"
zoo_model = Model(transfer_stub, download_path="./transfer_recipe")
recipe_path = zoo_model.recipes.default.path
```

### **Step 5: Setup Evaluation Metric**

We can use the [Evaluate](https://huggingface.co/docs/evaluate/index) library to compute and report metrics. 

```python
from transformers import EvalPrediction
from datasets import load_metric
import numpy as np

metric = load_metric("accuracy")

def compute_metrics(p: EvalPrediction):
  preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
  preds = np.argmax(preds, axis=1)
  return metric.compute(predictions=preds, references=p.label_ids)
```

### **Step 6: Train**

With the recipe created, we are now ready to kick off transfer learning. 

SparseML offers a custom `Trainer` class that inherits from the familiar [Hugging Face `Trainer`](https://huggingface.co/docs/transformers/main_classes/trainer). SparseML's `Trainer` extends the functionality to enable passing a `recipe` (such as the one we downloaded above). SparseML's `Trainer` parses the recipe and adjusts the training loop to apply the specified algorithms.

As you can see, it works just like Hugging Face's Trainer:

```python
from sparseml.transformers.sparsification import Trainer, TrainingArguments
from transformers import default_data_collator

# create TrainingArguments
training_args = TrainingArguments(
    output_dir="./training_output",
    do_train=True,
    do_eval=True,
    resume_from_checkpoint=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    save_total_limit=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    fp16=True)

# create SparseML Trainer
trainer = Trainer(
    model=model,
    model_state_path=model_path,
    recipe=recipe_path,
    teacher=teacher,
    metadata_args=["per_device_train_batch_size","per_device_eval_batch_size","fp16"],
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics)

# kick off training
train_result = trainer.train(resume_from_checkpoint=False)
trainer.save_model()  # Saves the tokenizer too for easy upload
trainer.save_state()
trainer.save_optimizer_and_scheduler(training_args.output_dir)
```

The `Trainer` class parses the transfer learning recipe and updates the training loop to maintain sparsity and apply quantization-aware training over the final epochs. After training for 13 epochs, the final model is 90% pruned and quantized reaching ~92% accuracy on the validation set.

Note that in this case, we passed the dense `teacher` we downloaded from SparseZoo to the `Trainer`. This is an optional argument (turn off by setting `teacher="disable"`), but can help to increase accuracy during the training process. You can specify the hyperparameters of the distillation process via the `DistillationModifiers`.

Checkout the `Trainer` and `TrainingArguments` API level docs for more details.

### **Step 7: Export to ONNX**

SparseML provides a `sparseml.transformers.export_onnx` command that you can use to export your trained model to ONNX. Be sure the `--model_path` argument points to your trained model:

```bash
sparseml.transformers.export_onnx \
 --model_path ./training_output \
 --task text_classification
```

The command creates a `./deployment` folder in your local directory, which contains the ONNX file and necessary Hugging Face tokenizer and configuration files for deployment with DeepSparse.

### **Other Examples**

Take a look at the tutorials for more examples in other use cases:

- [Sparse Transfer with GLUE Datasets (SST2) for sentiment analysis](sentiment-analysis/docs-sentiment-analysis-python-sst2.ipynb)
- [Sparse Transfer with Custom Datasets (RottenTomatoes) and Custom Teacher from HF Hub for sentiment analysis](sentiment-analysis/docs-sentiment-analysis-python-custom-teacher-rottentomatoes.ipynb)
- [Sparse Transfer with GLUE Datasets (QQP) for multi-input text classification](text-classification/docs-text-classification-python-qqp.ipynb)
- [Sparse Transfer with Custom Datasets (SICK) for multi-input text classification](text-classification/docs-text-classification-python-sick.ipynb)
- [Sparse Transfer with Custom Datasets (TweetEval) and Custom Teacher for single input text classification](text-classification/docs-text-classification-python-custom-teacher-tweeteval.ipynb)
- [Sparse Transfer with Custom Datasets (GoEmotions) for multi-label text classification](text-classification/docs-text-classification-python-multi-label-go_emotions.ipynb)
- [Sparse Transfer with Conll2003 for named entity recognition](token-classification/docs-token-classification-python-conll2003.ipynb)
- [Sparse Transfer with Custom Datasets (WNUT) and Custom Teacher for named entity recognition](token-classification/docs-token-classification-python-custom-teacher-wnut.ipynb)
- Sparse Transfer with SQuAD (example coming soon!)
- Sparse Transfer with Squadshifts Amazon (example coming soon!)
