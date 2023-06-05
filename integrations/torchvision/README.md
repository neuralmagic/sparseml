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

# SparseML Torchvision Integration

This directory explains how to use SparseML's `torchvision` integration to train inference-optimized sparse image classification models on your dataset.

There are two main workflows enabled by SparseML:

- **Sparse Transfer Learning** - fine-tune a pre-sparsified checkpoint on your own dataset **[RECOMMENDED]**
- **Sparsification from Scratch** - apply pruning and quantization to sparsify `torchvision` models from scratch

Once trained, SparseML enables you to export models to the ONNX format, such that they can be deployed with DeepSparse.

## Installation

Install with `pip`:

```bash
pip install sparseml[torchvision]
```

## Tutorials

- [Sparse Transfer Learning with the CLI - ResNet-50](tutorials/sparse-transfer-learning.md)
- [Sparse Transfer Learning with the Python API - ResNet-50](tutorials/docs-torchvision-python-transfer-imagenette.ipynb)
- [Sparsification from Scratch with the Python API - ResNet-50](tutorials/docs-torchvision-sparsify-from-scratch-resnet50-beans.ipynb)
- [Sparsification from Scratch with the Python API - MobileNetv2](tutorials/docs-torchvision-sparsify-from-scratch-mobilenetv2-beans.ipynb)

## Quick Tour

### SparseZoo

Neural Magic has pre-sparsified versions of common Torchvision models such as ResNet-50. These models can be deployed directly or can be fine-tuned onto custom dataset via sparse transfer learning. This makes it easy to create a sparse image classification model trained on your dataset.

[Check out the available models](https://sparsezoo.neuralmagic.com/?useCase=classification)

### Recipes

Recipes are YAML files that encode the instructions for sparsifying a model or sparse transfer learning. SparseML accepts the recipes as inputs, parses the instructions, and applies the specified algorithms and hyperparameters during the training process.

In such a way, recipes are the declarative interface for specifying which sparsity-related algorithms to apply.

### SparseML Python API

Because of the declarative, recipe-based approach, you can add SparseML to your PyTorch training pipelines with just a couple lines of code.

SparseML's `ScheduleModifierManager` class is responsible for parsing the YAML recipes and overriding the standard PyTorch model and optimizer objects, 
encoding the logic of the sparsity algorithms from the recipe. Once you call `manager.modify`, you can then use the model and optimizer as usual, as SparseML abstracts away the complexity of the sparsification algorithms.

The workflow looks like this:

```python
# typical model, optimizer, dataset definition
model = Model()
optimizer = Optimizer()
train_data = TrainData()

# parse recipe + edit model/optimizer with sparsity-logic
from sparseml.pytorch.optim import ScheduledModifierManager
manager = ScheduledModifierManager.from_yaml(PATH_TO_RECIPE)
optimizer = manager.modify(model, optimizer, len(train_data))

# PyTorch training loop, using the model/optimizer as usual

# clean-up
manager.finalize(model)
```

Note that the `model`, `optimizer`, and `dataset` are all standard PyTorch objects. We simply pass a recipe to the SparseML `SchedulerModifierManager` and SparseML handles the rest!

### SparseML CLI

In addition to the code-level API, SparseML offers pre-made training pipelines for image classification via the CLI interface.

The CLI enables you to kick-off training runs with various utilities like dataset loading and pre-processing, checkpoint saving, metric reporting, and logging handled for you.

To get started, we just need to a couple key arguments:
```bash
sparseml.image_classification.train \
    --checkpoint-path [CHECKPOINT-PATH] \
    --recipe [RECIPE-PATH] \
    --dataset-path [DATASET-PATH]
```

- `--checkpoint-path` specifies the starting model to use in the training process. It can either be a local path to a PyTorch checkpoint or a SparseZoo stub (which SparseML uses to download a PyTorch checkpoint).

- `--dataset-path` specifies the dataset used for training. It must be a local path to a dataset in the ImageNet format (see CLI tutorials for more details).

- `--recipe` specifies the sparsity related parameters of the training process. It can either be a local path to a YAML recipe file or a SparseZoo stub (which SparseML uses to download a YAML recipe file). The `recipe` is the key to enabling the sparsity-related algorithms implemented by SparseML (see the CLI tutorials for more details on recipes).

For full usage, run:
```bash
sparseml.image_classification.train --help
```

## Quick Start: Sparse Transfer Learning with the CLI

### Sparse Transfer Learning Overview

Sparse Transfer is quite similiar to the typical transfer learning process used to train image classification models, where we fine-tune a checkpoint pretrained on ImageNet onto a smaller downstream dataset. With Sparse Transfer Learning, we simply start the fine-tuning process from a pre-sparsified checkpoint and maintain sparsity while the training process occurs.

In this example, we will fine-tune a 95% pruned version of ResNet-50 ([available in SparseZoo](https://sparsezoo.neuralmagic.com/models/resnet_v1-50-imagenet-pruned95_quantized?comparison=resnet_v1-50-imagenet-base)) onto ImageNette.

### Kick off Training

We will use SparseML's `sparseml.torchvision.train` training script.

#### Sparse Transfer Recipe

To run sparse transfer learning, we first need to create/select a sparsification recipe. For sparse transfer, we need a recipe that instructs SparseML to maintain sparsity during training and to quantize the model.

For the Imagenette dataset, there is a transfer learning recipe available in SparseZoo, identified by the following SparseZoo stub:
```
zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none?recipe_type=transfer-classification
```

Here is what the recipe looks like:
   
```yaml
# Epoch and Learning-Rate variables
num_epochs: 10.0
init_lr: 0.0005

# quantization variables
quantization_epochs: 6.0

training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0.0
    end_epoch: eval(num_epochs)

  - !LearningRateFunctionModifier
    final_lr: 0.0
    init_lr: eval(init_lr)
    lr_func: cosine
    start_epoch: 0.0
    end_epoch: eval(num_epochs)

# Phase 1 Sparse Transfer Learning / Recovery
sparse_transfer_learning_modifiers:
  - !ConstantPruningModifier
    start_epoch: 0.0
    params: __ALL_PRUNABLE__

# Phase 2 Apply quantization
sparse_quantized_transfer_learning_modifiers:
  - !QuantizationModifier
    start_epoch: eval(num_epochs - quantization_epochs)
```
   
The key `Modifiers` for sparse transfer learning are the following:
- `ConstantPruningModifier` instructs SparseML to maintain the sparsity structure of the network during the fine-tuning process
- `QuantizationModifier` instructs SparseML to apply quantization aware training to quantize the weights over the final epochs

SparseML parses the `Modifers` in the recipe and updates the training loop with logic encoded therein.

#### Download Dataset

Download and unzip the Imagenette dataset:

```bash
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xvf imagenette2-320.tgz
```

#### Run the Training Script

Run the following to transfer learn from the 95% pruned-quantized ResNet-50
onto ImageNette:
```bash
sparseml.image_classification.train \
    --recipe zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none?recipe_type=transfer-classification \
    --checkpoint-path zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none?recipe_type=transfer-classification \
    --arch-key resnet50 \
    --dataset-path ./imagenette2-320 \
    --batch-size 32
```

The script uses the SparseZoo stubs to identify and download the starting checkpoint and YAML-based recipe file from the SparseZoo. SparseML parses the transfer learning recipe and adjusts the training logic to maintain sparsity during the fine-tuning process.

The resulting model is 95% pruned and quantized and is trained on ImageNette!

To transfer learn this sparsified model to other datasets you may have to adjust certain hyperparameters in this recipe and/or training script such as the optimizer type, the number of epochs, and the learning rates.

### Export to ONNX

The SparseML installation provides a `sparseml.image_classification.export_onnx` command that you can use to export the model to ONNX. Be sure the `--checkpoint_path` argument points to your trained model:

```bash
sparseml.image_classification.export_onnx \
  --arch_key resnet50 \
  --checkpoint_path ./checkpoint.pth \
  --dataset-path ./imagenette2-320
```

### Deploy with DeepSparse

Once exported to ONNX, you can deploy your models with DeepSparse. Checkout the DeepSparse repo for examples.
