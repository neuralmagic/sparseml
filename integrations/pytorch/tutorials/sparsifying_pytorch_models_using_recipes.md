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

# Sparsifying PyTorch Models Using Recipes

This tutorial shows how Neural Magic recipes simplify the sparsification process by encoding the hyperparameters and instructions needed to create highly accurate pruned and pruned-quantized PyTorch models.

## Overview

Neural Magic’s ML team creates recipes that allow anyone to plug in their data and leverage SparseML’s recipe-driven approach on top of their existing training pipelines.
Sparsifying involves removing redundant information from neural networks using algorithms such as pruning and quantization, among others.
This sparsification process results in many benefits for deployment environments, including faster inference and smaller file sizes.
Unfortunately, many have not realized the benefits due to the complicated process and number of hyperparameters involved.

Working through this tutorial, you will experience how Neural Magic recipes simplify the sparsification process by pruning a PyTorch [ResNet-50](https://arxiv.org/abs/1512.03385) model.

1. Setting Up
2. Inspecting a Recipe
3. Applying a Recipe
3. Exporting for Inference

## Need Help?

For Neural Magic Support, sign up or log in to our [**Deep Sparse Community Slack**](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue.](https://github.com/neuralmagic/sparseml/issues)

## Setting Up

This tutorial can be run by installing `sparseml` which contains scripts for this pruning example:

```bash
pip install "sparseml[torchvision, dev]"
```

## Inspecting a Recipe

Recipes are YAML or Markdown files that SparseML uses to easily define and control the sparsification of a model.
Recipes consist of a series of `Modifiers` that can influence the training process in different ways. A list of
common modifiers and their uses is provided
[here](../../../docs/source/recipes.md#modifiers-intro).

SparseML provides a recipe for sparsifying a ResNet-50 model trained on the tiny Imagenette dataset. The recipe can
be viewed in the browser
[here](../recipes/resnet50-imagenette-pruned.md)
or in the command line from the file `sparseml/integrations/pytorch/recipes/resnet50-imagenette-pruned.md`.

The recipe contains three kinds of modifiers:

- an `EpochRangeModifier` that sets the number of epochs used for sparsification
- a `SetLearningRateModifier` that sets the sparsification learning rate
- three `GMPruningModifier`s that set various layers of the model to be pruned to different levels.

Notice that each `GMPruningModifier` sets specific parameters of the ResNet-50 model to be pruned. To
prune a different model, the parameter names should be adjusted to match the new model.  To avoid naming
individual parameter names, all parameters can be set for pruning by passing `__ALL_PRUNABLE__` instead
of the parameters list for a single `GMPruningModifier`.

### Applying a Recipe

Recipes can integrated into training flows with a couple of lines of code by using a `ScheduledModifierManager`
that wraps the PyTorch `Optimizer` step.  An example of how this is done can be found
[here](../../../docs/source/code.md#pytorch-sparsification).

For this example, we can use the `sparseml.image_classification.train` utility.  This utility runs a
PyTorch training flow that is modified by a `ScheduledModifierManager` and takes a recipe as an input.
It natively supports ResNet-50 and other popular image classification models.

To prune a ResNet-50 model trained on the Imagenette dataset with the recipe above, run the following command.
If the Imagenette dataset has not been downloaded yet, the script will automatically download it to the given
`--dataset-path`. If you have a local copy of the dataset, `--dataset-path` may be updated.

*Note:* If an out of memory exception occurs, the train batch size should be lowered to fit on your device

```bash
sparseml.image_classification.train \
    --recipe-path integrations/pytorch/recipes/resnet50-imagenette-pruned.md \
    --dataset-path ./data \
    --pretrained True \
    --arch-key resnet50 \
    --dataset imagenette \
    --train-batch-size 128 \
    --test-batch-size 256 \
    --loader-num-workers 8 \
    --save-dir sparsification_example \
    --logs-dir sparsification_example \
    --model-tag resnet50-imagenette-pruned \
    --save-best-after 8
```

A full list of arguments and their descriptions can be found by running 

```bash
sparseml.image_classification.train --help
```


## Exporting for Inference

After, the training pruned model and training log files will be found under 
`sparsification_example/resnet50-imagenette-pruned`. To run the sparsified model for accelerated inference with
an inference engine such as [DeepSparse](https://github.com/neuralmagic/deepsparse), the pruned model must be
exported to the ONNX format.

This step can be completed using `sparseml.image_classification.export_onnx` script which uses the
`sparseml.pytorch.utils.ModuleExporter` class. Run the script with `--help` option to see usage.

The DeepSparse Engine is explicitly coded to support running sparsified models for significant improvements in
inference performance. An example for benchmarking and deploying image classification models with DeepSparse can
be found [here](https://github.com/neuralmagic/deepsparse/tree/main/examples/classification).

## Wrap-Up

Neural Magic recipes simplify the sparsification process by encoding the hyperparameters and instructions needed to create highly accurate sparsified models.
In this tutorial, you installed SparseML, downloaded and inspected a pruning recipe, applied it to a ResNet-50 model,
and exported the sparse model for inference.

Next steps will include creating new recipes and applying them to different models. Check out other
[SparseML tutorials](https://github.com/neuralmagic/sparseml/tree/main/integrations) to learn more.

For Neural Magic Support, sign up or log in to our [**Deep Sparse Community Slack**](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue.](https://github.com/neuralmagic/sparseml/issues)
