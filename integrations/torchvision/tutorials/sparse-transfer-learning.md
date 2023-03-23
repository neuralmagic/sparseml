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

# Sparsifying Image Classification Models with the CLI

This page explains how to fine-tune a pre-sparsified Torchvision image classification models like ResNet-50 with SparseML's CLI.

## Sparse Transfer Learning Overview

Sparse transfer learning is quite similiar to the typical transfer learning training, where we fine-tune a checkpoint pretrained on ImageNet onto a smaller downstream dataset. However, with Sparse Transfer Learning, we simply start the fine-tuning process from a pre-sparsified checkpoint and maintain sparsity while the training process occurs.

SparseZoo contains pre-sparsified checkpoints for common image classification models like ResNet-50 and EfficientNet, which can be used as the starting checkpoints for the training process.

[Check out the full list of pre-sparsified models](https://sparsezoo.neuralmagic.com/?domain=cv&sub_domain=classification&page=1)

## Installation

Install SparseML via `pip`.

```bash
pip install sparseml[torchvision]
```

## Sparse Transfer Learning with ResNet-50

Let's try a simple example of Sparse Transfer Learning ResNet-50 onto [FastAI's Imagenette dataset](https://github.com/fastai/imagenette) (a subset of 10 easily classified classes from ImageNet).

### Download Dataset

Download Imagnette to your local directory:
```bash
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xvf imagenette2-320.tgz
```

### Kick off Training

We will use SparseML's `sparseml.image_classification.train` training script.

#### Transfer Learning Recipe 

To run sparse transfer learning, we first need to create/select a sparsification recipe. For sparse transfer, we need a recipe that instructs SparseML to maintain sparsity during training and to quantize the model over the final epochs.

For ResNet-50, there is [premade transfer learning recipe available in SparseZoo](https://sparsezoo.neuralmagic.com/models/cv%2Fclassification%2Fresnet_v1-50%2Fpytorch%2Fsparseml%2Fimagenet%2Fpruned95_quant-none), identified by the following stub:
```bash
zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none?recipe_type=transfer-classification
```

Here's what the transfer learning recipe looks like:
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

The "Modifiers" encode how SparseML should modify the training process for Sparse Transfer Learning:
- `ConstantPruningModifier` tells SparseML to pin weights at 0 over all epochs, maintaining the sparsity structure of the network
- `QuantizationModifier` tells SparseML to quantize the weights with quantization aware training over the last 6 epochs

SparseML parses the instructions declared in the recipe and modifies the training loop accordingly before running the fine-tuning.

#### Run the Training Loop

Run the following to fine-tune the 95% pruned-quantized ResNet-50 on Imagenette.
```bash
sparseml.image_classification.train \
    --recipe zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none?recipe_type=transfer-classification \
    --checkpoint-path zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none?recipe_type=transfer-classification \
    --arch-key resnet50 \
    --dataset-path ./imagenette2-320\
    --batch-size 32
```

Let's discuss the key arguments:
- `--checkpoint-path` specifies the starting model to use in the training process. It can either be a local path to a PyTorch checkpoint or a SparseZoo stub. Here, we passed a stub that identifies a 95% pruned-quantized version of ResNet-50 in the SparseZoo. The script downloads the PyTorch model to begin training.

- `--arch-key` specifies the torchvision architecture of the checkpoint. For example, `resnet50` or `mobilenet`.

- `--dataset-path` specifies the dataset used for training. We passed the local path to Imagenette. The dataset should be in the ImageFolder format (we will describe the format below).

- `--recipe` specifies the transfer learning recipe. In this case, we passed a SparseZoo stub, which instructs SparseML to download a premade ResNet50 transfer learning recipe. In addition to SparseZoo stubs, you can also pass a local path to a YAML recipe.

The script uses the SparseZoo stubs to identify and download the starting checkpoint and recipe file. SparseML then parses the transfer learning recipe and adjusts the training loop to maintain sparsity during the fine-tuning process. It then kicks off the transfer learning run.

The resulting model is 95% pruned and quantized ResNet-50 trained on Imagenette!

Run the help command to inspect the full list of arguments and configurations.
```bash
sparseml.image_classification.train --help
```

### Export to ONNX

Run the following to export your model to ONNX for deployment with DeepSparse.

```bash
sparseml.image_classification.export_onnx \
  --arch_key resnet50 \
  --checkpoint_path ./checkpoint.pth \
  --dataset-path ./imagenette2-320
```

### Modify a Recipe

To transfer learn this sparsified model to other datasets you may have to adjust certain hyperparameters in this recipe and/or training script. Some considerations:

- For more complex datasets, increase the number of epochs, adjusting the learning rate step accordingly
- Adding more learning rate step milestones can lead to more jumps in accuracy
- Increase the learning rate when increasing batch size
- Increase the number of epochs if using SGD instead of the Adam optimizer
- Update the base learning rate based on the number of steps needed to train your dataset

To update a recipe, you can download the YAML file from SparseZoo, make updates to the YAML directly, and pass the local path to SparseML.

Alternatively, you can use `--recipe_args` to modify a recipe on the fly. The following runs for 15 epochs instead of 10:

```bash
sparseml.image_classification.train \
    --recipe zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none?recipe_type=transfer-classification \
    --recipe_args '{"num_epochs": 15}' \
    --checkpoint-path zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none?recipe_type=transfer-classification \
    --arch-key resnet50 \
    --dataset-path ./imagenette2-320\
    --batch-size 32
```

The `--recipe_args` are a json parsable dictionary of recipe variable names to values to overwrite the YAML-recipe.

### Dataset Format

SparseML's Torchvision CLI conforms to the [Torchvision ImageFolder dataset format](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html), where images are arranged into directories representing each class. To use a custom dataset with the SparseML CLI, you will need to arrange your data into this format.

For example, the following downloads Imagenette to your local directory:

```bash
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xvf imagenette2-320.tgz
```

The resulting `imagenette-320` directory looks like the following (where each subdirectory like `n01440764` 
represents one of the 10 classes in the Imagenette dataset) and each `.JPEG` file is a training example.
```bash
|-train
    |-n01440764
        |-n01440764_10026.JPEG
        |-n01440764_10027.JPEG
        |...
    |-n02979186
    |-n03028079
    |-n03417042
    |-n03445777
    |-n02102040
    |-n03000684
    |-n03394916
    |-n03425413
    |-n03888257
|-val
    |-n01440764
    |-n02979186
    | ...
```

## Wrapping Up

Checkout the DeepSparse repository for more information on deploying your sparse Image Classification models with DeepSparse for GPU class performance on CPUs.
