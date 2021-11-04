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

This directory combines the SparseML recipe-driven approach with the
[huggingface/transformers](https://github.com/huggingface/transformers) repository.
By integrating the robust training flows in the `transformers` repository with the SparseML code base,
we enable model sparsification techniques on popular NLP models such as [BERT](https://arxiv.org/abs/1810.04805)
creating smaller and faster deployable versions.
The techniques include, but are not limted to:
- Pruning
- Quantization
- Pruning and Quantization
- Sparse Transfer Learning

## Highlights

Coming soon!

## Tutorials

- [Sparsifying BERT Models Using Recipes](https://github.com/neuralmagic/sparseml/blob/main/integrations/huggingface-transformers/tutorials/sparsifying_bert_using_recipes.md)

## Installation

To begin, run the following command in the root directory of this integration (`cd integrations/huggingface-transformers`):
```bash
bash setup_integration.sh
```

The `setup_integration.sh` file will clone the transformers repository with the SparseML integration as a subfolder.
After the repo has successfully cloned, transformers and datasets will be installed along with any necessary dependencies.

It is recommended to run Python 3.8 as some of the scripts within the transformers repository require it.

## Quick Tour

Recipes encode the instructions and hyperparameters for sparsifying a model using modifiers to the training process.
The modifiers can range from pruning and quantization to learning rate and weight decay.
When appropriately combined, it becomes possible to create highly sparse and accurate models.

This integration adds a `--recipe` argument to the [`run_qa.py`](https://github.com/neuralmagic/transformers/blob/master/examples/pytorch/question-answering/run_qa.py) script among others.
The argument loads an appropriate recipe while preserving the rest of the training pipeline.
Popular recipes used with this argument are found in the [`recipes` folder](./recipes).
Distillation arguments to support student-teacher distillation are additionally added to the scripts as they help improve the recovery while sparsifying.
Otherwise, all other arguments and functionality remain the same as the original repository.

For example, pruning and quantizing a model on the SQuAD dataset can be done by running the following command from within the root of this integration's folder:
```bash
python transformers/examples/pytorch/question-answering/run_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --evaluation_strategy epoch \
  --per_device_train_batch_size 16 \
  --learning_rate 5e-5 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir MODELS_DIR/bert-base-12layers_prune80 \
  --cache_dir cache \
  --preprocessing_num_workers 6 \
  --fp16 \
  --num_train_epochs 30 \
  --recipe recipes/bert-base-12layers_prune80.md \
  --onnx_export_path MODELS_DIR/bert-base-12layers_prune80/onnx \
  --save_strategy epoch \
  --save_total_limit 2
```

### Structure

The following table lays out the root-level files and folders along with a description for each.

| Folder/File Name     | Description                                                                                                           |
|----------------------|-----------------------------------------------------------------------------------------------------------------------|
| recipes              | Typical recipes for sparsifying NLP models along with any downloaded recipes from the SparseZoo.                      |
| tutorials            | Tutorial walkthroughs for how to sparsify NLP models using recipes.                                                   |
| transformers         | Integration repository folder used to train and sparsify NLP models (`setup_integration.sh` must run first).            |
| README.md            | Readme file.                                                                                                          |
| setup_integration.sh | Setup file for the integration run from the command line.                                                             |

### Exporting for Inference

After sparsifying a model, the `run_qa.py` script can be run with the `--onnx_export_path` argument to convert the model into an [ONNX](https://onnx.ai/) deployment format.
The export process is modified such that the quantized and pruned models are corrected and folded properly.

For example, the following command can be run from within the integration's folder to export a trained/sparsified model's checkpoint:
```bash
python transformers/examples/pytorch/question-answering/run_qa.py \
  --model_name_or_path MODELS_DIR/bert-base-12layers_prune80 \
  --dataset_name squad \
  --do_eval \
  --per_device_eval_batch_size 64 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir MODELS_DIR/bert-base-12layers_prune80/eval \
  --cache_dir cache \
  --preprocessing_num_workers 6 \
  --onnx_export_path MODELS_DIR/bert-base-12layers_prune80/onnx
```

The DeepSparse Engine [accepts ONNX formats](https://docs.neuralmagic.com/sparseml/source/onnx_export.html) and is engineered to significantly speed up inference on CPUs for the sparsified models from this integration.
Examples for loading, benchmarking, and deploying can be found in the [DeepSparse repository here](https://github.com/neuralmagic/deepsparse).
