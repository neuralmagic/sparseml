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

# Sparse Transfer Learning for Image Classification

This tutorial shows how Neural Magic sparse models simplify the sparsification process by offering pre-sparsified models for transfer learning onto other datasets.

## Overview

Neural Magic’s ML team creates sparsified models that allow anyone to plug in their data and leverage pre-sparsified models from the SparseZoo. 
Sparsifying involves removing redundant information from neural networks using algorithms such as pruning and quantization, among others. 
This sparsification process results in many benefits for deployment environments, including faster inference and smaller file sizes. 

Unfortunately, many have not realized the benefits due to the complicated process and number of hyperparameters involved.
Working through this tutorial, you will experience how Neural Magic recipes simplify the sparsification process. In this tutorial you will:
- Download and prepare a pre-sparsified image classification model.
- Apply a sparse transfer learning recipe on the pre-sparsified model.

The examples listed in this tutorial are all performed on the [Imagenette](https://github.com/fastai/imagenette) dataset.

## Need Help?

For Neural Magic Support, sign up or log in to our [**Deep Sparse Community Slack**](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue.](https://github.com/neuralmagic/sparseml/issues) 

## Setting Up

This tutorial can be run by cloning and installing the `sparseml` repository which contains scripts and recipes for
this example:

```bash
pip install "sparseml[torchvision, dev]"
```
Note:  make sure to upgrade `pip` using `python -m pip install -U pip`
## Downloading and Preparing a Pre-Sparsified Model

First, you need to download the sparsified models from the [SparseZoo](https://sparsezoo.neuralmagic.com/). A few image classification models with [SparseZoo](https://sparsezoo.neuralmagic.com/) stubs:

| Model Name     |      Stub      | Description |
|----------|-------------|-------------|
| resnet-pruned-moderate | zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned-moderate |This model is a sparse, [ResNet-50](https://arxiv.org/abs/1512.03385) model that achieves 99% of the accuracy of the original baseline model (76.1% top1). Pruned layers achieve 88% sparsity.|
|resnet-pruned-conservative|zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned-conservative|This model is a sparse, [ResNet-50](https://arxiv.org/abs/1512.03385) model that achieves full recovery original baseline model accuracy (76.1% top1). Pruned layers achieve 80% sparsity.|
|resnet-pruned-moderate-quantized|zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned_quant-moderate|This model is a sparse, INT8 quantized [ResNet-50](https://arxiv.org/abs/1512.03385) model that achieves 99% of the original baseline accuracy. The average model sparsity is about 72% with pruned layers achieving between 70-80% sparsity. This pruned quantized model achieves 75.46% top1 accuracy on the ImageNet dataset.|
|mobilenetv1-pruned-moderate|zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate|This model is a sparse, [MobileNetV1](https://arxiv.org/abs/1704.04861) model that achieves 99% of the accuracy of the original baseline model (70.9% top1). Pruned layers achieve between 70-90% sparsity.|
|mobilenetv1-pruned-conservative|zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-conservative|This model is a sparse, [MobileNetV1](https://arxiv.org/abs/1704.04861) model that achieves the same accuracy as the original baseline model. Pruned layers achieve between 60-86% sparsity. This pruned quantized model achieves 70.9% top1 accuracy on the ImageNet dataset.|
|mobilenetv1-pruned-moderate-quantized|zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned_quant-moderate|Pruned [MobileNetV1](https://arxiv.org/abs/1704.04861) architecture with width=1.0 trained on the ImageNet dataset. Recalibrated performance model: within 99% of baseline validation top1 accuracy (70.9%)|


A few recipe stubs:

|Recipe Name|Stub|Description|
|----------|-------------|-------------|
|resnet50.transfer-learning-pruned-quantized|zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned_quant-moderate?recipe_type=transfer_learn|This recipe provides a framework for performing pruned-quantized transfer learning on SparseML's pruned_quantized ResNet50 model. This recipe was written for transfer learning onto the Imagenette dataset using a single GPU, Adam optimizer, and batch size 32. Adjust the hyperparameters at the top of the file accordingly for your dataset and training environment|


Note: The models above were originally trained and sparsified on the [ImageNet](https://image-net.org/) dataset.

- After deciding on which model meets your performance requirements for both speed and accuracy, select the SparseZoo stub associated with that model.
   The stub will be used later with the training script to automatically pull down the desired pre-trained weights.

## Applying a Sparse Transfer Learning Recipe

- After noting respective SparseZoo model stub, [train.py](../train.py) script can be used to download checkpoint and [Imagenette](https://github.com/fastai/imagenette) and kick-start transfer learning. 
   The transfer learning process itself is guided using recipes; We include example [recipes](../recipes) for classification along with others in the SparseML [GitHub repository](https://github.com/neuralmagic/sparseml). 
[Learn more about recipes and modifiers](https://github.com/neuralmagic/sparseml/tree/main/docs/source/recipes.md).

- Run the following example command to kick off transfer learning for [ResNet-50](https://arxiv.org/abs/1512.03385) starting from a moderately pruned checkpoint from [SparseZoo](https://sparsezoo.neuralmagic.com/):
```
sparseml.image_classification.train \
    --recipe-path integrations/pytorch/recipes/classification.transfer_learn_pruned.md \
    --checkpoint-path "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned-moderate" \
    --arch-key resnet50 \
    --model-kwargs '{"ignore_error_tensors": ["classifier.fc.weight", "classifier.fc.bias"]}' \
    --dataset imagenette \
    --dataset-path /PATH/TO/IMAGENETTE  \
    --train-batch-size 32 --test-batch-size 64 \
    --loader-num-workers 0 \
    --optim Adam \
    --optim-args '{}' \
    --model-tag resnet50-imagenette-sparse-transfer-learned
```

To transfer learn [MobileNet](https://arxiv.org/abs/1704.04861) on [Imagenette](https://github.com/fastai/imagenette), starting from a moderately pruned checkpoint from [SparseZoo](https://sparsezoo.neuralmagic.com/), run the following example command:

```
sparseml.image_classification.train \
    --recipe-path integrations/pytorch/recipes/classification.transfer_learn_pruned.md \
    --checkpoint-path "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate" \
    --arch-key mobilenet \
    --model-kwargs '{"ignore_error_tensors": ["classifier.fc.weight", "classifier.fc.bias"]}' \
    --dataset imagenette \
    --dataset-path /PATH/TO/IMAGENETTE  \
    --train-batch-size 32 --test-batch-size 64 \
    --loader-num-workers 0 \
    --optim Adam \
    --optim-args '{}' \
    --save-dir sparse-transfer-models
    --model-tag mobilenet-imagenette-sparse-transfer-learned
```
The `--checkpoint-path` argument can take in path to a model checkpoint, or a SpareZoo model stub,
if a SparseZoo recipe stub is provided in `--recipe-path`, 
the `checkpoint-path` can be set to `zoo` for auto-downloading corresponding checkpoints from SparseZoo. 
To learn more about the usage, run script with `-h` option to see help. 

The script automatically saves model checkpoints after each epoch and reports the validation loss along with layer sparsities. 

### ONNX Export
The model can be saved to [ONNX](https://onnx.ai/) format to be loaded later for inference or other experiments using
the `sparseml.image_classification.export_onnx` script.

```
sparseml.image_classification.export_onnx \
    --arch-key mobilenet \
    --dataset imagenette \
    --dataset-path /PATH/TO/IMAGENETTE \
    --checkpoint-path sparse-transfer-models/mobilenet-imagenette-sparse-transfer-learned/pytorch/model.pth
```

## Wrap-Up

Neural Magic sparse models and recipes simplify the sparsification process by enabling sparse transfer learning to create highly accurate pruned image classification models. 
In this tutorial, you downloaded a pre-sparsified model, applied a Neural Magic recipe for sparse transfer learning, and saved the model for future use.

An example for benchmarking and deploying image classification models with DeepSparse [is also available](https://github.com/neuralmagic/deepsparse/tree/main/examples/classification).

For Neural Magic Support, sign up or log in to our [**Deep Sparse Community Slack**](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue.](https://github.com/neuralmagic/sparseml/issues)
