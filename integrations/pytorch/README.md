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

# SparseML PyTorch Integration

This directory combines the SparseML recipe-driven approach with standard PyTorch training flows for models within
[torchvision](https://pytorch.org/vision/stable/index.html) and SparseML.
They are intended as example flows to show how to integrate sparsification techniques with a custom PyTorch training flow.
The techniques include, but are not limited to:
- Pruning
- Quantization
- Pruning + Quantization
- Sparse Transfer Learning

## Highlights

- Blog: [ResNet-50 on CPUs: Sparsifying for Better Performance on CPUs](https://neuralmagic.com/blog/benchmark-resnet50-with-deepsparse/)

## Tutorials

- [Classification](./notebooks/classification.ipynb)
- [Detection](./notebooks/detection.ipynb)
- [Sparse Quantized Transfer Learning](./notebooks/sparse_quantized_transfer_learning.ipynb)
- [Torchvision](./notebooks/torchvision.ipynb)

## Installation

To begin, run `pip install sparseml[torchvision]`

## Quick Tour

### Recipes

Recipes encode the instructions and hyperparameters for sparsifying a model using modifiers to the training process.
The modifiers can range from pruning and quantization to learning rate and weight decay.
When appropriately combined, it becomes possible to create highly sparse and accurate models.

This integration contains `--recipe` arguments appended to the Python scripts and `recipe` variables in the appropriate notebooks.
Popular recipes used with this argument are found in the [`recipes` folder](./recipes).

### SparseZoo

Pre-sparsified models and recipes can be downloaded through the [SparseZoo](https://github.com/neuralmagic/sparsezoo).

Complete lists are available online for all [models](https://sparsezoo.neuralmagic.com/tables/models/cv/classification?repo=sparseml&framework=pytorch) and 
[recipes](https://sparsezoo.neuralmagic.com/tables/recipes/cv/classification?repo=sparseml&framework=pytorch) compatible with this integration as well.

Sample code for retrieving a model from the SparseZoo:
```python
from sparsezoo import Model

model = Model("zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned_quant-moderate")
print(model)
```

Sample code for retrieving a recipe from the SparseZoo:
```python
from sparsezoo import Model

model = Model("zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned_quant-moderate")
recipe = model.recipes.default
print(recipe)
```

### Structure

The following table lays out the root-level files and folders along with a description for each.

| Folder/File Name                                             | Description                                                                                                           |
|--------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| [notebooks](./notebooks)                                     | Jupyter notebooks to walk through sparsifiation within standard PyTorch training flows                                |
| [recipes](./recipes)                                         | Typical recipes for sparsifying PyTorch models along with any downloaded recipes from the SparseZoo.                  |
| [tutorials](./tutorials)                                     | Tutorial walkthroughs for how to sparsify PyTorch models using recipes.                                               |
| [README.md](./README.md)                                     | Readme file.                                                                                                          |
| [torchvision_sparsification.py](./torchvision_sparsification.py) | Example training script for sparsifying PyTorch torchvision classification models.                                    |
| `sparseml.image_classification.train`                        | Utility training script for sparsifying classification and detection models in PyTorch (loads SparseML models).       |
| `sparseml.image_classification.export`                       | Utility script for exporting pytorch model to ONNX file along with sample inputs, outputs and labels.                 |
| `sparseml.image_classification.lr_analysis`                  |Utility script to Run learning rate sensitivity analysis for a desired image classification architecture |`sparseml.image_classification.pr_sensitivity`                |Utility script to Run a kernel sparsity (pruning) analysis for a desired image classification architecture |

### Exporting for Inference

All scripts and notebooks end with an export to the [ONNX](https://onnx.ai/) file format.
The export process is modified such that the quantized and pruned models are corrected and folded properly.

The DeepSparse Engine accepts ONNX formats and is engineered to significantly speed up inference on CPUs for the sparsified models from this integration.
Examples for loading, benchmarking, and deploying can be found in the [DeepSparse repository here](https://github.com/neuralmagic/deepsparse/).
