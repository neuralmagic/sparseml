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

By integrating with robust training flows in the Torchvision repository, SparseML enables you to train inference-optimized sparse versions of popular image classification models like ResNet-50 on your dataset.

There are two pathways:
- **Sparse Transfer Learning**: fine-tune a pre-sparsified checkpoint on your own dataset **[RECOMMENDED]**
- **Sparsification from Scratch**: Apply state-of-the-art training-aware pruning and quantization algorithms to arbitrary Torchvision models.

Once trained, SparseML enables you to export models to the ONNX format, such that they can be deployed with DeepSparse for GPU-class performance on the CPU.

## Installation

Install with `pip`:

```bash
pip install sparseml[torchvision]
```

## Tutorials

- [Torchvision CLI](tutorials/torchvision-cli.md)
- [Sparse Transfer Learning with the Python API](tutorials/docs-torchvision-python-transfer-imagenette.ipynb)
- Sparsification from Scratch with the Python API (coming soon!)

## Quick Tour

### SparseZoo

Neural Magic has pre-sparsified versions of common Torchvision models such as ResNet-50. These models can be deployed directly or can be fine-tuned onto custom dataset via sparse transfer learning. This makes it easy to create a sparse image classification model trained on your dataset.

Check out the model cards in the [SparseZoo](https://sparsezoo.neuralmagic.com/?repo=ultralytics&page=1).

### Recipes

SparseML Recipes are YAML files that encode the instructions for sparsifying a model or sparse transfer learning. The SparseML image classification training script accepts the recipes as inputs, parses the instructions, and applies the specified algorithms and hyperparameters during the training process.

### SparseML CLI

SparseML's CLI enables you to kick-off sparsification workflows with various utilities like creating training pipelines, dataset loading, checkpoint saving, metric reporting, and logging handled for you.

To get started, we just need to a couple key arguments: a starting checkpoint, a SparseML recipe, and a dataset.

```bash
sparseml.image_classification.train \
    --checkpoint-path [CHECKPOINT-PATH] \
    --recipe [RECIPE-PATH] \
    --dataset-path [DATASET-PATH]
```

The key arguments are as follows:
- `--checkpoint-path` specifies the starting model to use in the training process. It can either be a local path to a PyTorch checkpoint or a SparseZoo stub (which SparseML uses to download a PyTorch checkpoint).

- `--dataset-path` specifies the dataset used for training. It must be a local path to a dataset in the ImageNet format (see CLI tutorials for more details).

- `--recipe` specifies the sparsity related parameters of the training process. It can either be a local path to a YAML recipe file or a SparseZoo stub (which SparseML uses to download a YAML recipe file). The `recipe` is the key to enabling the sparsity-related algorithms implemented by SparseML (see the CLI tutorials for more details on recipes).

For full usage, run:
```bash
sparseml.image_classification --help
```

### SparseML Python API

For additional flexibility, SparseML has a Python API that also enables you to add sparsification to a native PyTorch training loop.

Just like the CLI, the Python API uses YAML-based recipes to encode the parameters of the sparsification process, allowing youto add SparseML with just a few lines of code.

The `ScheduleModifierManager` class is responsible for parsing the YAML recipes and overriding the standard PyTorch model and optimizer objects, 
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

See the tutorials for full working examples of the Python API.

## Quick Start: Sparse Transfer Learning

### Overview

Sparse Transfer is quite similiar to the typical transfer learing process used to train NLP models, where we fine-tune a pretrained checkpoint onto a smaller downstream dataset. With Sparse Transfer Learning, we simply start the fine-tuning process from a pre-sparsified checkpoint and maintain sparsity while the training process occurs.

In this example, we will fine-tune a 95% pruned version of ResNet-50 ([available in SparseZoo](https://sparsezoo.neuralmagic.com/models/cv%2Fclassification%2Fresnet_v1-50%2Fpytorch%2Fsparseml%2Fimagenet%2Fpruned95_quant-none)) onto ImageNette.

### Kick off Training

We can start Sparse Transfer Learning by passing a starting checkpoint and recipe to the training script. For Sparse Transfer, we will use a recipe that instructs SparseML to maintain sparsity during training and to quantize the model. The 95% pruned-quantized ResNet-50 has a transfer learning recipe available, identified by the following SparseZoo stub:
```
zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none?recipe_type=transfer-classification
```

<details>
   <summary>Click to see the recipe</summary>

SparseML parses the `Modifers` in the recipe and updates the training loop with logic encoded therein.
   
The key `Modifiers` for sparse transfer learning are the following:
- `ConstantPruningModifier` instructs SparseML to maintain the sparsity structure of the network during the fine-tuning process
- `QuantizationModifier` instructs SparseML to apply quantization aware training to quantize the weights over the final epochs
   
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

</details>

First, download and unzip the ImageNette dataset:

```bash
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xvf imagenette2-320.tgz
```

Run the following to transfer learn from the 95% pruned-quantized ResNet-50
onto ImageNette:
```bash
sparseml.image_classification.train \
    --recipe zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none?recipe_type=transfer-classification \
    --checkpoint-path zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none?recipe_type=transfer-classification \
    --arch-key resnet50 \
    --dataset-path ./imagenette2-320\
    --batch-size 32
```

The script uses the SparseZoo stubs to identify and download the starting checkpoint and YAML-based recipe file from the SparseZoo. SparseML parses the transfer learning recipe and adjusts the trainign process to maintain sparsity during the fine-tuning process.

The resulting model is 95% pruned and quantized, and achieves 99% validation accuracy on ImageNette!

To transfer learn this sparsified model to other datasets you may have to adjust certain hyperparameters in this recipe and/or training script. Some considerations:
- For more complex datasets, increase the number of epochs, adjusting the learning rate step accordingly
- Adding more learning rate step milestones can lead to more jumps in accuracy
- Increase the learning rate when increasing batch size
- Increase the number of epochs if using SGD instead of the Adam optimizer
- Update the base learning rate based on the number of steps needed to train your dataset

### Export to ONNX

DeepSparse uses the ONNX format to load neural networks and then deliver breakthrough performance for CPUs by leveraging the sparsity and quantization within a network.

The SparseML installation provides a `sparseml.image_classification.export_onnx` command that you can use to export the model to ONNX. Be sure the `--weights` argument points to your trained model. Be sure the `--checkpoint_path` argument points to your trained model:

```bash
sparseml.image_classification.export_onnx \
  --arch_key resnet50 \
  --checkpoint_path ./checkpoint.pth \
  --dataset-path ./imagenette2-320
```

### Deploy with DeepSparse

Once exported to ONNX, you can deploy your models with DeepSparse. Checkout the DeepSparse repo for examples.
