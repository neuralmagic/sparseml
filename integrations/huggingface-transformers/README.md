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

# SparseML Hugging Face Transformers Integration

This directory combines the SparseML recipe-driven approach with the [huggingface/transformers](https://github.com/huggingface/transformers) repository. By integrating the robust training flows in the `transformers` repository with the SparseML code base, we enable model sparsification techniques on popular NLP models such as [BERT](https://arxiv.org/abs/1810.04805) creating smaller and faster deployable versions. The techniques include, but are not limited to:

- Pruning
- Quantization
- Knowledge Distillation
- Sparse Transfer Learning

## Highlights

Coming soon!

## Tutorials

- [Sparsifying BERT Models Using Recipes](https://github.com/neuralmagic/sparseml/blob/main/integrations/huggingface-transformers/tutorials/sparsifying_bert_using_recipes.md)
- [Sparse Transfer Learning With BERT](https://github.com/neuralmagic/sparseml/blob/main/integrations/huggingface-transformers/tutorials/bert_sparse_transfer_learning.md)

## Installation

```bash
pip install sparseml[torch]
```

It is recommended to run Python 3.8 as some of the scripts within the transformers repository require it.

**Note**: Transformers will not immediately install with this command. Instead, a sparsification-compatible version of Transformers will install on the first invocation of the Transformers code in SparseML.

## SparseML CLI

The SparseML installation provides a CLI for sparsifying your models for a specific task; appending the `--help` argument will provide a full list of options for training in SparseML:

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

## Sparse Transfer Learning | Question Answering Example

### Dense Teacher Creation

To enable distillation, you will first create a dense teacher model that the sparse model will learn from while transferring. **If you already have a Transformers-compatible model, you can use this as the dense teacher in place of training one from scratch.** The following command will use the dense BERT base model from the SparseZoo and fine-tune it on the SQuAD dataset, resulting in a model that achieves 88.5% F1 on the validation set: 

```bash
sparseml.transformers.question_answering \
    --output_dir models/teacher \
    --model_name_or_path zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/base-none \
    --recipe zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/base-none?recipe_type=transfer-question_answering \
    --dataset_name squad \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 24 \
    --preprocessing_num_workers 6 \
    --do_train \
    --do_eval \
    --evaluation_strategy epoch \
    --fp16 \
    --seed 42 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 24 \
    --save_strategy epoch \
    --save_total_limit 1
```

With the dense teacher trained to convergence, you can begin the sparse transfer learning with distillation with a recipe. The dense teacher will distill knowledge into the sparse architecture, therefore increasing its performance while ideally converging to the dense solution’s accuracy.

💡**PRO TIP**💡: Recipes encode the instructions and hyperparameters for sparsifying a model using modifiers to the training process. The modifiers can range from pruning and quantization to learning rate and weight decay. When appropriately combined, it becomes possible to create highly sparse and accurate models.

Once the command has completed, you will have a sparse checkpoint located in `models/sparse_quantized`.

### Transfer Learn the Model

The following command will use the 80% sparse-quantized BERT model from the SparseZoo and fine-tune it on the SQuAD dataset, resulting in a model that achieves an F1 of 88.5% on the validation set. Keep in mind that the `--distill_teacher` argument is set to pull a dense SQuAD model from the SparseZoo to enable it to run independent of the dense teacher step. If you trained a dense teacher, change this out for the path to your model folder:

```bash
sparseml.transformers.question_answering \
    --output_dir models/sparse_quantized \
    --model_name_or_path zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/12layer_pruned80_quant-none-vnni \
    --recipe zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/12layer_pruned80_quant-none-vnni?recipe_type=transfer-question_answering \
    --distill_teacher zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none \
    --dataset_name squad \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 24 \
    --preprocessing_num_workers 6 \
    --do_train \
    --do_eval \
    --evaluation_strategy epoch \
    --fp16 \
    --seed 21636  \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 24 \
    --preprocessing_num_workers 6 \
    --save_strategy epoch \
    --save_total_limit 1
```

### Exporting to ONNX

The DeepSparse Engine uses the ONNX format to load neural networks and then deliver breakthrough performance for CPUs by leveraging the sparsity and quantization within a network.

The SparseML installation provides a `sparseml.transformers.export_onnx` command that you can use to load the training model folder and create a new model.onnx file within. Be sure the `--model_path` argument points to your trained model. By default, it is set to the result from transfer learning a sparse-quantized BERT model:

```bash
sparseml.transformers.export_onnx \
    --model_path models/sparse_quantized \
    --task 'question-answering' \
    --sequence_length 384
```

### DeepSparse Engine Deployment

Now that the model is in an ONNX format, it is ready for deployment with the DeepSparse Engine. 

Run the following command to install it:

```bash
pip install deepsparse
```

Once DeepSparse is installed on your deployment environment, two options are supported for deployment: 
- A Python API that will fit into our current deployment pipelines.
- The DeepSparse Server that enables a no-code CLI solution to run your model via FastAPIs HTTP server.

### 🐍 Python API

The Python code below gives an example for using the DeepSparse Python pipeline API with different tasks. Be sure to change out the `model_path` argument for the model folder of your trained model:

Python Pipeline:

```python
from deepsparse import Pipeline

model_path = "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/12layer_pruned80_quant-none-vnni"

qa_pipeline = Pipeline.create(
  task="question-answering", 
  model_path=model_path
)

inference = qa_pipeline(question="What's my name?", context="My name is Snorlax")
print(inference)
```
printout:

    {'score': 0.9947717785835266, 'start': 11, 'end': 18, 'answer': 'Snorlax'}

### 🔌DeepSparse Server

To use the DeepSparse Server, first install the required dependencies using pip:

```bash
pip install deepsparse[server]
```

Once installed, the CLI command given below for serving a BERT model is available. The commands are set up to be able to run independently of the prior stages. Once launched, you can view info over the server and the available APIs at `http://0.0.0.0:5543` on the deployment machine. 

```bash
deepsparse.server \
    --task question_answering \
    --batch_size 1 \
    --model_path "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/12layer_pruned80_quant-none-vnni"
```

For more details, check out the [Getting Started with the DeepSparse Server](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/server).