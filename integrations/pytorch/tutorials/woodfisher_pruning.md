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

# WoodFisher Pruning with SparseML

This tutorial shows how to use the WoodFisher pruning algorithm in a SparseML PyTorch
pruning recipe. The WoodFisher algorithm was introduced in the paper 
[WoodFisher: Efficient Second-Order Approximation for Neural Network Compression](https://arxiv.org/pdf/2004.14340.pdf)
by Sidak Pal Singh and Dan Alistarh.

## Overview
The WoodFisher algorithm introduces an efficient process for approximating accurate
second-order information about a model's weights that can be used to solve for optimal
weights to prune.  The method uses the weights' first-order information
(gradients) to approximate the Inverse-Hessian matrix that is used to score weights
for pruning.

This method was shown to achieve SoTA results for pruning tasks on
image classification models such as ResNet and MobileNet. Detailed information of the
methods and results can be found in the [paper](https://arxiv.org/pdf/2004.14340.pdf).

WoodFisher pruning can be enabled by a single modifier in a SparseML recipe. This tutorial
will demonstrate how to adapt an existing SparseML pruning recipe to do this.

## WoodFisher Pruning Recipes
To use the WoodFisher algorithm in a SparseML pruning recipe, the `MFACPruningModifier`
must be used in place of the existing pruning modifier(s).  the `MFACPruningModifier`
contains the same parameters as SparseML's magnitude pruning modifiers for controlling
epoch range, sparsity levels, and schedule, so these values may remain
unchaged when adjusting a recipe.

The `MFACPruningModifier` implements a global pruning scheme however, so all
parameters to be pruned should be listed under the `params` argument. If the previous
recipe has multiple pruning modifiers, all of their prameters should be merged into a
single `MFACPruningModifier`.

WoodFisher specific hyperparameters can be specified under the `mfac_options` parameter.
Values should be listed as a YAML dictionary.  Full detials about these hyperparameters
are listed in the next section of this tutorial.

### Example Recipe Section

The following is an example `MFACPruningModifier` to be used in place of other
PruningModifiers in a recipe:

```yaml
pruning_modifiers:

  - !MFACPruningModifier
    params: __ALL_PRUNABLE__
    init_sparsity: 0.05
    final_sparsity: 0.85
    start_epoch: 1.0
    end_epoch: 61.0
    update_frequency: 4.0
    mfac_options:
      num_grads: {0.0: 256, 0.5: 512, 0.75: 1024, 0.83: 1400}
      fisher_block_size: 10000
      available_gpus: ["cuda:0"]
```

## WoodFisher Parameters
The following parameters can be specified under the `mfac_options` parameter to control
how the WoodFisher calculations are made. Ideal values will depend on the system
available to run on and model to be pruned.

### num_grads
To approximate the second order information in the WoodFisher algorithm, first order
gradients are used. `num_grads` specifies the number of recent gradient samples to store
of a model while training.

This value can be an int where that constant value will be used throughout pruning or a
dictionary of float sparsity values to the number of gradients that should be
stored when that sparsity level (between 0.0 and 1.0) is reached. If a
dictionary, then 0.0 must be included as a key for the base number of gradients
to store (i.e. {0: 64, 0.5: 128, 0.75: 256}).

Storing gradients can be expensive, as for a dense model, each additional gradient
sample stored requires about the same memory that the entire model needs. This is why
the dictionary option allows for more gradients to be stored as the model gets more
sparse.

If a WoodFisher pruning run is unexpectedly killed, the reason could likely be that
the gradient storage requirements exceeded the system's RAM. A safe rule of thumb for
initial number of gradients is the number should be no greater than 1/4 of the
available CPU RAM divided by the model size.


### fisher_block_size
To limit the computational cost of calculating second order information, the WoodFisher
algorithm may compute a block diagonal matrix of a certain block size that is
sufficient for generating the necessary information for pruning.

The `fisher_block_size` specifies this block size.  If using GPUs to perform the
WoodFisher computations, the GPUs should have `num_grads * fisher_block_size` extra
memory duing training so each block can be stored and computed sequentially on a GPU.

The default block size is 2000, and generally block sizes between 1000 and 10000 may be
ideal. If `None` is provided, the full matrix will be computed without blocks.


### available_gpus
`available_gpus` is a list of GPU devices names to perform the WoodFisher computation
with. If not provided, computation will be done on the CPU.


## Wrap-Up
This tutorial shows how to use WoodFisher pruning by swapping in a single modifier
to a SparseML pruning recipe. Everything else for pruning and exporting stays the same.
More papers and results will be posted as they become available.

Open an
[issue](https://github.com/neuralmagic/sparseml/issues) with any questions or bugs.