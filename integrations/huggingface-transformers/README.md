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

This directory demonstrates how to use SparseML's Hugging Face `transformers` integration. 

By integrating the robust training flows in the `transformers` repository, SparseML enables you to create sparse versions of popular NLP models such as [BERT](https://arxiv.org/abs/1810.04805) trained on your dataset. The techniques include, but are not limited to:
- Pruning
- Quantization
- Knowledge Distillation
- Sparse Transfer Learning

Once trained, SparseML enables you to export models to the ONNX format - such that they can be deployed with DeepSparse.

## Tutorials

- XXX

## Installation

```bash
pip install sparseml[torch]
```

**Note**: Transformers will not immediately install with this command. Instead, a sparsification-compatible version of Transformers will install on the first invocation of the Transformers code in SparseML.

## SparseML CLI

SparseML's CLI enables you to kick-off sparsification workflows with various utilities like dataset loading, checkpoint saving, metric reporting, and logging handled for you; appending the `--help` argument will provide a full list of options for training in SparseML:

```bash
sparseml.transformers.[task] --help
```

e.g. `sparseml.transformers.question_answering --help`

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

## SparseML Python API

Alternatively, SparseML offers a custom `Trainer` class that inherits from `transformers`'s [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer). 

Beyond the native `transformers` functionality, the SparseML `Trainer` also accepts a `recipe` argument, which is a path to a local YAML file
containing a configuration with sparsity-related algorithms and hyperparameters. The `Trainer` class parses the recipe and makes the adjustments
to the training process to apply sparsification algorithms or sparse transfer learning to the model. 

As such, you can apply sparsification algorithms from within the `transformers` framework, leveraging the friendly utilities such as 
`AutoConfigs`, `AutoTokenizers`, `AutoModels`, and `datasets` as well as the Hugging Face Hub. For instance, in the example below, 
the `model`, `teacher`, `dataset`, `tokenizer`, and `compute_metrics` are all native `transformers` objects. Becuase of the 

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

## Sparse Transfer Learning Example - Sentiment Analysis

### Overview

Sparse Transfer Learning is the best pathway for creating a sparse model trained on your dataset/task. Sparse Transfer is quite similiar to the typical
training process used to train NLP models for downstream tasks, where we start with a checkpoint trained via masked language modeling on an upstream task 
like WikipediaBookCorpus and fine-tune it onto a smaller dataset with a specific task. However, with Sparse Transfer Learning, we start the training 
process from a pre-sparsified checkpoint and **maintain sparsity** while the training process occurs.

In our case, there is a [90% pruned version of BERT available in SparseZoo](https://sparsezoo.neuralmagic.com/models/nlp%2Fmasked_language_modeling%2Fobert-base%2Fpytorch%2Fhuggingface%2Fwikipedia_bookcorpus%2Fpruned90-none). It is identified by the following SparseZoo stub:

```
zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none
```

In this example, we will fine-tune the 90% pruned version of BERT onto the SST2 dataset.

### Dense Teacher Creation

To support the transfer learning process, we can optionally apply model distillation.

To enable distillation, we first create a dense teacher model. **If you already have a Transformers-compatible model, you can use this as the dense teacher in place of training one from scratch.** The following command will use the dense BERT base model from the SparseZoo and fine-tune it on the SS%2 dataset, resulting in a model that achieves 92.9% accuracy on the validation set: 

```bash
sparseml.transformers.text_classification \
  --output_dir dense_obert-text_classification_sst2 \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/base-none \
  --recipe zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/base-none \
  --task_name sst2 --max_seq_length 128 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --preprocessing_num_workers 6 \
  --do_train --do_eval --evaluation_strategy epoch --fp16 --seed 20811 \
  --save_strategy epoch --save_total_limit 1
```

Once the command has completed, you will have a sparse checkpoint located in `dense_obert-text_classification_sst2`.

### Transfer Learn the Model

The following command will use the 90% pruned BERT model from the SparseZoo and fine-tune it on the SST2 dataset, resulting in a model that achieves an accuracy of 92% on the validation set. Keep in mind that the `--distill_teacher` argument is set to pull a dense SST2 model from the SparseZoo. If you trained a dense teacher with the command from above, update the script to use `--distill_teacher ./dense_obert-text_classification_sst2`.

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

### Exporting to ONNX

The DeepSparse Engine uses the ONNX format to load neural networks and then deliver breakthrough performance for CPUs by leveraging the sparsity and quantization within a network.

The SparseML installation provides a `sparseml.transformers.export_onnx` command that you can use to load the training model folder and create a new model.onnx file within. Be sure the `--model_path` argument points to your trained model:

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

Python Pipeline:

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