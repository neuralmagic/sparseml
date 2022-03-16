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

# Gradual Pruning with M-FAC

This tutorial shows how to aggressively prune an example MobileNet model using
both magnitude and [Matrix-Free Approximate Curvature (M-FAC)](https://arxiv.org/pdf/2107.03356.pdf)
gradual pruning.

## Overview
Magnitude pruning selects weights with the smallest magnitudes for removal from a model.
This technique provides a strong standard for model pruning that can prune models to reasonably
high sparsities with minimal accuracy degradation.
Instead of magnitudes, the M-FAC pruning algorithm efficiently uses first-order model
information (gradients) to approximate second-order information that can be used to
decide which weights are optimal to prune. This technique can often lead to superior
results to magnitude pruning when pruning to higher sparsities or when pruning with fewer
epochs.

In this tutorial, you will prune a MobileNet model trained on the tiny Imagenette dataset - 
once with magnitude pruning and once with M-FAC pruning.  Both pruning runs will
prune the model to 95% sparsity and only use 3 epochs. The selected model, dataset, and
pruning schedule are selected for the purposes of this tutorial only to quickly demonstrate
the benefits of the M-FAC algorithm.  In practice, a longer pruning schedule and larger dataset
should be used.


### Steps
1. Setting Up
2. Inspecting the Recipes
3. Running M-FAC and Magnitude Pruning
4. Comparing the Results


## Need Help?

For Neural Magic Support, sign up or log in to our [**Deep Sparse Community Slack**](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue.](https://github.com/neuralmagic/sparseml/issues)

## Setting Up

This tutorial can be run by cloning and installing the `sparseml` repository which contains scripts and recipes for
this pruning example:

```bash
git clone https://github.com/neuralmagic/sparseml.git
pip install sparseml[torchvision]
```

## Inspecting the Recipes

To launch both M-FAC and magnitude pruning, you will need two separate recipes.
For this tutorial, the recipes for pruning MobileNet to 95% sparsity in 3 epochs are provided
in SparseML. They can be viewed with the command line or by following the links below:

* M-FAC: [sparseml/research/mfac/recipes/pruning-mobilenet-imagenette-mfac-short-95.md](https://github.com/neuralmagic/sparseml/blob/main/research/mfac/recipes/pruning-mobilenet-imagenette-mfac-short-95.md)
* Magnitude: [sparseml/research/mfac/recipes/pruning-mobilenet-imagenette-magnitude-short-95.md](https://github.com/neuralmagic/sparseml/blob/main/research/mfac/recipes/pruning-mobilenet-imagenette-magnitude-short-95.md)

Notice that the recipes are identical except that the `MFACPruningModifier` and `mfac_options`
are used in M-FAC pruning.  The M-FAC recipe's `available_gpus` parameter should be updated
based on your system's available devices. If no GPU is available, then the line may be deleted.

## Running M-FAC and Magnitude Pruning

SparseML provides an integrated training script that can be used to sparsify common
computer vision models. The following commands can be used to prune MobileNet
trained on the Imagenette dataset with the recipes above.  Both training runs
should take a couple minutes with the M-FAC run taking a little longer depending on
the available hardware.

```bash
cd sparseml
```

M-FAC:
```bash
python integrations/pytorch/vision.py train \
    --recipe-path research/mfac/recipes/pruning-mobilenet-imagenette-mfac-short-95.md \
    --dataset-path ./data \
    --pretrained True \
    --arch-key mobilenet \
    --dataset imagenette \
    --train-batch-size 32 \
    --test-batch-size 64 \
    --loader-num-workers 8 \
    --save-dir mfac_pruning_example \
    --logs-dir mfac_pruning_example \
    --model-tag mobilenet-pruned-mfac \
    --save-best-after 2
```

Magnitude:
```bash
python integrations/pytorch/vision.py train \
    --recipe-path research/mfac/recipes/pruning-mobilenet-imagenette-magnitude-short-95.md \
    --dataset-path ./data \
    --pretrained True \
    --arch-key mobilenet \
    --dataset imagenette \
    --train-batch-size 32 \
    --test-batch-size 64 \
    --loader-num-workers 8 \
    --save-dir mfac_pruning_example \
    --logs-dir mfac_pruning_example \
    --model-tag mobilenet-pruned-magnitude \
    --save-best-after 2
```


## Comparing the Results

After both training runs complete, you can compare the results:

```bash
# M-FAC results
cat mfac_pruning_example/mobilenet-pruned-mfac/model.txt
```

```bash
Magnitude results
cat mfac_pruning_example/mobilenet-pruned-magnitude/model.txt
```

For M-FAC, you should see that the final top-1 accuracy recovered to
~88.6% while magnitude recovers to ~86.6%.  So in this example, M-FAC
beats magnitude by 2% in Top-1 accuracy recovery.  In a pruning
scenario with a larger dataset and pruning schedule, the results may
differ.  When selecting which pruning algorithm to use it is the recovery
as well as the training time should be considered.

## Wrap-Up
In this tutorial you applied both M-FAC and magnitude pruning with SparseML and compared
their results. More information about M-FAC pruning and other tutorials can be found
[here](https://github.com/neuralmagic/sparseml/blob/main/research/mfac).

For Neural Magic Support, sign up or log in to our [**Deep Sparse Community Slack**](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue.](https://github.com/neuralmagic/sparseml/issues)
