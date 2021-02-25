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

# PyTorch Sparse-Quantized Transfer Learning with SparseML

<figure class="image">
  <img src="https://docs.neuralmagic.com/docs/source/sparsification/flow-sparsification-model_recipe_sparsezoo-sparseml_transfer_learning.svg" max-width="960">
</figure>

## Overview
[Pruning](https://neuralmagic.com/blog/pruning-overview/) and
[quantization](https://arxiv.org/abs/1609.07061) are well-established methods for accelerating
neural networks.  Individually, both methods yield significant speedups for CPU inference
(a theoretical maximum of 4x for INT8 quantization) and can make CPU deployments an attractive
option for real-time model inference.

Sparse-quantized models leverage both techniques and
[can achieve speedups upwards of 7x](https://neuralmagic.com/blog/benchmark-resnet50-with-deepsparse)
when using the [DeepSparse Engine](https://github.com/neuralmagic/deepsparse) with
[compatible hardware](https://docs.neuralmagic.com/deepsparse/hardware.html).

Using powerful [SparseML](https://github.com/neuralmagic/sparseml) recipes, it is easy to create sparse-quantized models.
Additionally, the SparseML team is actively creating pre-trained sparse-quantized models that maintain accuracy
targets and achieve high CPU speedups - and it is easy to leverage these models for speedups with your own datasets
using sparse-quantized transfer learning.

## Sparse-Quantized Transfer Learning

[Transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) is a technique that
involves retraining a pre-trained model to learn a new task, with the benefit of starting
from the pre-trained model's already learned behavior. Sparse-quantized transfer takes the
additional step of reusing both the pre-trained weights, and pre-trained sparse model
structure of an existing sparse-quantized model to train it on a new ML task.

This technique allows engineers and researchers to create sparse-quantized optimizations
for one model and then easily re-apply them to accelerate many tasks.

Sparse-quantized transfer learning takes place in two phases:
1. Sparse transfer learning \- fine tuning the pre-trained model with the new dataset
while maintaining the existing pre-optimized sparsity structure.  This creates a model 
that learns to predict a new task, while preserving the predetermined optimized structure
from pruning.
2. [Quantization-aware training](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/#quantization-aware-training)
\- emulating the effects of INT8 quantization while training the model to overcome the loss of precision


## ResNet-50 Imagenette Example

The [SparseZoo](https://github.com/neuralmagic/sparseml) hosts a sparse-quantized ResNet-50 model trained
on the ImageNet dataset.  It maintains 99% of the baseline accuracy and can achieve over 6.5x
speedup using the DeepSparse Engine.  There are multiple paths to explore sparse-quantized
transfer learning with this model.

### Notebook
`sparseml/examples/pytorch_sparse_quantized_transfer_learning/pytorch_sparse_quantized_transfer_learning.ipynb`
is a Jupyter Notebook that provides a step-by-step walk-through for
 - setting up sparse-quantized transfer learning
 - integrating SparseML with any PyTorch training flow
 - ONNX export
 - benchmarking with the DeepSparse Engine 

Run `jupyter notebook` and navigate to this notebook file to run the example.

### Script
`sparseml/scripts/pytorch_vision.py` is a script for running tasks related to pruning and
quantization with SparseML for image classification and object detection use cases.
Using the following example command, you can run sparse-quantized transfer learning on a custom
[ImageFolder](https://pytorch.org/vision/0.8/datasets.html#imagefolder) based
classification dataset.

Note that for datasets other than Imagenette, you may need to edit
the recipe to better train for the dataset following instructions in the downloaded recipe card.

```
python scripts/pytorch_vision.py train \
    --recipe-path zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned_quant-moderate?recipe_type=transfer_learn \
    --checkpoint-path zoo \
    --arch-key resnet50 \
    --model-kwargs '{"ignore_error_tensors": ["classifier.fc.weight", "classifier.fc.bias"]}' \
    --dataset imagefolder \
    --dataset-path /PATH/TO/IMAGEFOLDER/DATASET  \
    --train-batch-size 32 --test-batch-size 64 \
    --loader-num-workers 8 \
    --optim Adam \
    --optim-args '{}' \
    --model-tag resnet50-imagenette-pruned_quant-transfer_learned
```


### Further Reading
To learn more about this sparse-quantized ResNet-50 model's benchmarks check out
[this blog post](https://neuralmagic.com/blog/benchmark-resnet50-with-deepsparse).

For more information on creating sparse and quantized models, check out our
[documentation](https://neuralmagic.com/contact/).

SparseML and its companion packages are open-source and constantly improving.
Keep an eye out for new models, algorithms, and faster speeds.
If you have any questions, extensions, or use cases feel free to,
[contribute](https://github.com/neuralmagic/sparseml/blob/main/CONTRIBUTING.md),
[open an issue](https://github.com/neuralmagic/sparseml/issues),
or [contact us](https://neuralmagic.com/contact/).