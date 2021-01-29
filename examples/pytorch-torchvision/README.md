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

# SparseML-torchvision integration
This directory demonstrates how to integrate SparseML optimizations with the [torchvision](https://pytorch.org/docs/stable/torchvision/index.html)
library and provides an ease-of-use script for doing so.

Some of the tasks you can perform using this integration include, but are not limited to:
* model pruning
* sparse quantization-aware-training
* sparse model fine-tuning and transfer learning

Reading through the notebook or script, you will also be able to see how to easily integrate a generic
PyTorch training flow with SparseML.

## Installation
To begin, run `pip install sparseml[torchvision]`

## Notebook
For a quick, step-by-step walk-through of performing the integration and pruning a model run through the
[pruning.ipynb](https://github.com/neuralmagic/sparseml/blob/main/examples/pytorch-torchvision/pruning.ipynb) notebook.

Run `jupyter notebook` in your terminal and navigate to the notebook in your browser to get started.

## Script
`examples/pytorch-torchvision/main.py` is an ease-of-use script for applying a SparseML optimization recipe to a torchvision classification model.
The script file is fully documented with descriptions, a command help printout, and example commands.
You can also run `python examples/pytorch-torchvision/main.py -h` for a help printout.

To run this script, you will need a SparseML recipe as well as an
[ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder)-like classification dataset to train
your model with.

You can learn how to build or download a recipe using the
[SparseML](https://github.com/neuralmagic/sparseml)
or [SparseZoo](https://github.com/neuralmagic/sparsezoo)
documentation, or export one with [Sparsify](https://github.com/neuralmagic/sparsify).

The [Imagenette](https://github.com/fastai/imagenette) dataset can be used as an initial
dataset if you need one.

To run the script you will need to specify the model name as stored in
[`torchvision.models`](https://pytorch.org/docs/stable/torchvision/models.html),
the path to your recipe, and path to your dataset.  Optional parameters including batch size, pretrained and others can be found
in the script file documentation.  The optimization learning rate, and number of epochs are set in your SparseML recipe.

example command:
```bash
python examples/pytorch-torchvision/main.py \
    --recipe-path ~/sparseml_recipes/pruning_resnet50.yaml \
    --model resnet50 \
    --imagefolder-path ~/datasets/my_imagefolder \
    --batch-size 128
```  
