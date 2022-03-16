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

# Matrix-Free Approximate Curvature (M-FAC) Pruning

The paper
[Efficient Matrix-Free Approximations of Second-Order Information, with Applications to Pruning and Optimization](https://arxiv.org/pdf/2107.03356.pdf)
written by Elias Frantar, Eldar Kurtic, and Assistant Professor Dan Alistarh of IST Austria
introduces the Matrix-Free Approximate Curvature (M-FAC) method of pruning.
M-FAC builds on advances from the [WoodFisher](https://arxiv.org/pdf/2004.14340.pdf)
pruning paper to efficiently use first-order information (gradients) to determine optimal weights
to prune by approximating the corresponding second-order information.
This algorithm is shown to outperform magnitude pruning as well as other second-order pruning
techniques on a variety of one-shot and gradual pruning tasks.

## Using M-FAC with SparseML

SparseML makes it easy to use the M-FAC pruning algorithm as part of sparsification
recipes to improve pruning recovery by providing an `MFACPruningModifier`.
The `MFACPruningModifier` contains the same settings as the magnitude
pruning modifiers and contains extra settings for the M-FAC algorithm including 
`num_grads`, `fisher_block_size`, and `available_gpus`. Ideal values will depend 
on the system available to run on and model to be pruned.

### Example M-FAC Recipe
The following is an example `MFACPruningModifier` to be used in place of other
pruning modifiers in a recipe:

```yaml
pruning_modifiers:
  - !MFACPruningModifier
    params: __ALL_PRUNABLE__
    init_sparsity: 0.05
    final_sparsity: 0.85
    start_epoch: 1.0
    end_epoch: 61.0
    update_frequency: 4.0
    num_grads: {0.0: 256, 0.5: 512, 0.75: 1024, 0.83: 1400}
    fisher_block_size: 10000
    available_gpus: ["cuda:0"]
```

#### num_grads
To approximate the second order information in the M-FAC algorithm, first order
gradients are used. `num_grads` specifies the number of recent gradient samples to store
of a model while training.

This value can be an int where that constant value will be used throughout pruning.
Alternatively the value can be a dictionary of float sparsity values to the number of
gradients that should be stored when that sparsity level (between 0.0 and 1.0) is reached.
If a dictionary is used, then 0.0 must be included as a key for the base number of gradients
to store (i.e. {0: 64, 0.5: 128, 0.75: 256}).

Storing gradients can be expensive, as for a dense model, each additional gradient
sample stored requires about the same memory that the entire model needs. This is why
the dictionary option allows for more gradients to be stored as the model gets more
sparse.

If a M-FAC pruning run is unexpectedly killed, the reason could likely be that
the gradient storage requirements exceeded the system's RAM. A safe rule of thumb for
initial number of gradients is the number should be no greater than 1/4 of the
available CPU RAM divided by the model size.


#### fisher_block_size
To limit the computational cost of calculating second order information, the M-FAC
algorithm may compute a block diagonal matrix of a certain block size that is
sufficient for generating the necessary information for pruning.

The `fisher_block_size` specifies this block size.  If using GPUs to perform the
M-FAC computations, the GPUs should have `num_grads * fisher_block_size` extra
memory during training so each block can be stored and computed sequentially on a GPU.

The default block size is 2000, and generally block sizes between 1000 and 10000 may be
ideal. If `None` is provided, the full matrix will be computed without blocks.


#### available_gpus
`available_gpus` is a list of GPU devices names to perform the WoodFisher computation
with. If not provided, computation will be done on the CPU.


## Tutorials

Tutorials for using M-FAC with SparseML are provided in the [tutorials](https://github.com/neuralmagic/sparseml/blob/main/research/mfac/tutorials)
directory.  Currently there are tutorials available for
[one-shot](https://github.com/neuralmagic/sparseml/blob/main/research/mfac/tutorials/one_shot_pruning_with_mfac.md)
and [gradual](https://github.com/neuralmagic/sparseml/blob/main/research/mfac/tutorials/gradual_pruning_with_mfac.md)
pruning with M-FAC.

## Need Help?
For Neural Magic Support, sign up or log in to our [**Deep Sparse Community Slack**](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue.](https://github.com/neuralmagic/sparseml/issues)
