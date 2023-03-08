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

# Recipes

Recipes are SparseML's declarative interface for specifying sparsity-related algorithms that should be applied during training.
The SparseML system parses the recipes and modifies the model and/or training loop with the encoded instructions.

This allows you to add SparseML's algorithms to your existing training loops with a few lines of code.

## Modifiers

SpareseML recipes are YAML files or markdown files with YAML front matter.

The recipes contain a list of `modifiers`, which are the objects that specify how SparseML should update the training process. There are four types of modifiers:
- [`training_modifiers`](#training-modifiers) specify hyperparameters of the training process, such as the learning rate schedule or number of epochs
- [`pruning_modifiers`](#pruning-modifiers) specify how pruning should be applied to a model, such as the level of sparsity and pruning algorithms
- [`quantization_modifiers`](#quantization-modifiers) specify how quantization-aware training should be applied to a model, such as which layers to target
- [`distillation_modifiers`](#distillation-modifiers) specify how model distillation should be applied during training, such as which layers to target and hyperparameters like hardness and temperature

A sample recipe for pruning a model generally looks like the following:

```yaml
lr_func: constant
init_lr: 0.001
final_lr: 0.0
num_epochs: 20.0
num_pruning_active_epochs: 10.0
num_pruning_finetuning_epochs: 10.0
pruning_init_sparsity: 0.05
pruning_final_sparsity: 0.8
pruning_update_frequency: 0.5
mask_type: unstructured
global_sparsity: True
_num_pruning_epochs: eval(num_pruning_active_epochs + num_pruning_finetuning_epochs)

training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0.0
    end_epoch: eval(num_epochs)

  - !LearningRateFunctionModifier
    start_epoch: 0.0
    end_epoch: eval(num_epochs)
    lr_func: eval(lr_func)
    init_lr: eval(init_lr)
    final_lr: eval(final_lr)

pruning_modifiers:
  - !GlobalMagnitudePruningModifier
    start_epoch: 0.0
    params: __ALL_PRUNABLE__
    init_sparsity: 0.05
    final_sparsity: 0.8
    end_epoch: eval(num_pruning_active_epochs)
    update_frequency: 1.0
    mask_type: eval(mask_type)
    leave_enabled: True
    inter_func: cubic
```

[Check out the SparseZoo](https://sparsezoo.neuralmagic.com/) to see examples of state-of-the-art recipes/

Below, we enumerate the available modifiers in SparseML.

## Training Modifiers

### Hyperparameter Modifiers

The hyperparameter modifiers control various aspects of the training process.

#### EpochRangeModifier

The `EpochRangeModifier` is a simple modifier to control the range of epochs for training a model. Note that if other modifiers exceed the range of `EpochRangeModifier` for min or max epochs, this modifier will not have an effect.

Parameters:
- `start_epoch`: The start range for the epoch (0 indexed)
- `end_epoch`: The end range for the epoch

 Example:

 ```yaml
num_epochs: 25.0
training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0.0
    end_epoch: eval(num_epochs)
 ```

[Source Code](https://github.com/neuralmagic/sparseml/blob/main/src/sparseml/pytorch/sparsification/training/modifier_epoch.py)

#### LearningRateFunctionModifier

The `LearningRateFunctionModifier` sets the learning rate (LR) for the optimizer based on support math functions scaling between an `init_lr` and a `final_lr`. If you are using an Adam optimizer, then generally, these are not useful. If you are using a standard stochastic gradient descent optimizer, then these give a convenient way to control the LR.

Parameters:
- `lr_func`: The name of the learning rate function to use (options: `linear`, `cosine`, `cyclic_linear`)
- `init_lr`: The initial learning rate to use at `start_epoch`
- `final_lr`: The final learning rate to use at `end_epoch`
- `start_epoch`: The epoch to start the modifier at (set to -1.0 so it starts immediately)
- `end_epoch`: The epoch to end the modifier at (set to -1.0 so it doesn't end)
- `cycle_epoch`: The number of epochs between two consecutive LR rewinding (used for `cyclic_linear` schedule only)
- `groups`: The param group indices to set the lr for within the optimizer, if not set will set the lr for all param groups

Example:

```yaml
lr_func: constant
init_lr: 0.001
final_lr: 0.0
num_epochs: 20.0

training_modifiers:
  - !LearningRateFunctionModifier
    start_epoch: 0.0
    end_epoch: eval(num_epochs)
    lr_func: eval(lr_func)
    init_lr: eval(init_lr)
    final_lr: eval(final_lr)
```

[Source Code](https://github.com/neuralmagic/sparseml/blob/main/src/sparseml/pytorch/sparsification/training/modifier_lr.py#L223)

#### SetWeightDecayModifier

The `SetWeightDecayModifier` sets the weight decay (L2 penalty) for the optimizer to a specific value.

Parameters:

- `start_epoch`: The epoch in the training process to set the `weight_decay` value for the optimizer.
- `weight_decay`: The floating-point value to set as the weight decay for the optimizer at `start_epoch`.
- `param_groups`: The indices of param groups in the optimizer to be modified. If None, all param groups will be modified. Default: None.

Example:

```yaml
training_modifiers:
   - !SetWeightDecayModifier
      start_epoch: 5.0
      weight_decay: 0.01
```

[Source Code](https://github.com/neuralmagic/sparseml/blob/main/src/sparseml/pytorch/sparsification/training/modifier_regularizer.py)

## Pruning Modifiers

The pruning modifiers handle [pruning](https://neuralmagic.com/blog/pruning-overview/). Pruning is the process of removing weight connections in a network to increase inference speed and decrease model storage size. In general, neural networks are very over parameterized. Pruning a network can be thought of as removing unused parameters from the over parameterized network.

SparseML implements several pruning-related algorithms, which are uses to achieve the two main workflows:
- For **Sparse Transfer Learning**, where we fine-tune a pre-sparsified model onto a downstream dataset, the `ConstantPruningModifier` instructs SparseML to maintain sparsity as the model fine-tunes..
- For **Sparsifying From Scratch**, where we prune weights from a dense network, there are many options in SparseML. We recommend starting with [`GlobalMagnitudePruningModifier`](#globalmagnitudepruningmodifier) or [`OBSPruningModifier`](#obspruningmodifier). With Transformers models espeically, OBS is a great choice.

### Constant Pruning

Constant pruning modifiers are used to maintain sparsity while fine-tuning occurs. This is useful for Sparse Transfer Learning, where we fine-tune a pre-sparsified model (such as the sparse checkpionts in Neural Magic's SparseZoo) onto a new task.

#### ConstantPruningModifier

`ConstantPruningModifier` enforces the sparsity structure and level for an already pruned layer(s) in a model. The non-zero weights remain trainable in this setup but weights that are 0 are pinned to 0, maintaining the sparsity structure of the network.

Required Parameters:
- `start_epoch`: The epoch to start the modifier at
- `end_epoch`: The epoch to end the modifier at
- `params`:  A list of full parameter names or regex patterns of names to apply the modifier to.  Regex patterns must be specified with the prefix 're:'. `__ALL__` will match to all parameters. `__ALL_PRUNABLE__` will match to all ConvNd and Linear layers' weights. You will generally want to use `__ALL_PRUNABLE__`.

Example:

```yaml
training_modifiers:
   - !ConstantPruningModifier
     start_epoch: 0.0
     params: __ALL_PRUNABLE__
```

[Source](https://github.com/neuralmagic/sparseml/blob/main/src/sparseml/pytorch/sparsification/pruning/modifier_pruning_constant.py)

### Gradual Magnitude Pruning

The [gradual magnitude pruning algorithm (GMP)](https://neuralmagic.com/blog/pruning-gmp/) is used to sparsify a dense networks from scratch. GMP is a relatively simple, but effective algorithm. It works by starting with a pretrained network and fine-tuning non-zero weights further. At the end of each epoch, the weights closest to 0 are removed from the network. This process is repeated until the target level of sparsity is reached. 

In such a way, we gradually induce sparsity into the network, while fine-tuning the non-zero weights to help the model recover to the new optimization space. 

#### GlobalMagnitudePruningModifier

`GlobalMagnitudePruningModifier` applies the GMP algorithm with global sparsity. That is, layers are pruned globally, such that average sparsity across layers is equal to `target_sparsity` (but each layer may have different sparsity levels). If applying GMP, you should default to global pruning over layerwise, especially for vision models.

Parameters:
- `init_sparsity`: Initial sparsity for each layer at `start_epoch`
- `final_sparsity`: Final sparisty for each layer at `end_epoch`
- `start_epoch`: The epoch to start the GMP algorithm
- `end_epoch`: The epoch to end the GMP algorithm
- `update_frequency`: Frequency at which weights are pruned
- `params`: A list of full parameter names or regest patters to apply pruning to. Regex pattens must be specified with the prefix `re:` (e.g. `"re:.*weight"`). `__ALL__` will match all parameters. `__ALL_PRUNABLE__` will match to all ConvNd and Linear layers' weights
- `inter_func`: The type of interpolation function to use, which determines how sparsity is added to the network. Options are `linear`, `cubic`, and `inverse_cubic`. `cubic` is the default
- `mask_type`: String to define the type of sparsity to apply. Options are `unstructured` for unstructured pruning or `4block` for four block pruning or a list of two integers for a custom block shape. In almost all cases, you should use `unstructured`
- `leave_enabled`: is a boolean value which specifies whether to maintain sparsity after the `end_epoch`. This should generally be set to `True` (the default)

Example:

```yaml
num_pruning_epochs: 10.0
mask_type: unstructured
inter_func: cubic

training_modifiers:
   - !MagnutidePruningModifier
     init_sparsity: 0.05
     final_sparsity: 0.8
     start_epoch: 0.0
     end_epoch: eval(num_pruning_epochs)
     update_frequency: 1.0
     params: __ALL_PRUNABLE__
     leave_enabled: True
     inter_func: eval(inter_func)
     mask_type: eval(mask_type)
```

#### MagnitudePruningModifier

`MagnitudePruningModifier` applies the GMP algorithm with layerwise sparsity. That is, each layer is pruned independently such that the final sparsity for each layer equals the `target_sparsity`.

Parameters:
- `init_sparsity`: Initial sparsity for each layer at `start_epoch`
- `final_sparsity`: Final sparisty for each layer at `end_epoch`
- `start_epoch`: The epoch to start the GMP algorithm
- `end_epoch`: The epoch to end the GMP algorithm
- `update_frequency`: Frequency at which weights are pruned
- `params`: A list of full parameter names or regest patters to apply pruning to. Regex patersn must be specified with the prefix `re:` (e.g. `"re:.*weight"`). `__ALL__` will match all parameters. `__ALL_PRUNABLE__` will match to all ConvNd and Linear layers' weights
- `inter_func`: The type of interpolation function to use, which determines how sparsity is added to the network. Options are `linear`, `cubic`, and `inverse_cubic`. `cubic` is the default
- `mask_type`: String to define the type of sparsity to apply. Options are `unstructured` for unstructured pruning or `4block` for four block pruning or a list of two integers for a custom block shape. In almost all cases, you should use `unstructured`

Example:

```yaml
num_pruning_epochs: 10.0
mask_type: unstructured
inter_func: cubic

training_modifiers:
   - !MagnutidePruningModifier
     init_sparsity: 0.05
     final_sparsity: 0.8
     start_epoch: 0.0
     end_epoch: eval(num_pruning_epochs)
     update_frequency: 1.0
     params: __ALL_PRUNABLE__
     leave_enabled: True
     inter_func: eval(inter_func)
     mask_type: eval(mask_type)
```

[Source Code](https://github.com/neuralmagic/sparseml/blob/main/src/sparseml/pytorch/sparsification/pruning/modifier_pruning_magnitude.py)

### First and Second Order Methods

Beyond gradual magnitude pruning, where weights that are closest to 0 are iteratively pruned at the end of every epoch, more advanced techniques have been developed to use first and second order methods to identify the least impactful weights. These methods are generally more compute intensive (as they can require approximating the Hessian matrix), but are helpful for reaching higher levels of sparsity without accuracy loss. 

#### OBSPruningModifier

`OBSPruningModifier` implements the [OBS algorithm](https://arxiv.org/abs/2203.07259). Just like gradual magnitude pruning, the OBS algorithm iteratively prunes the least important weights at the end of each epoch. However, rather than removing weights closest to 0 (like GMP), the OBS algorithm is a second-order method that approximates the Hessian matrix to identify the least important weights. While OBS is compute-intensive, it is a generally a great place to start when pruning transformers models (Neural Magic uses this method internally).

Parameters:
- `init_sparsity`: Initial sparsity for each layer at `start_epoch`
- `final_sparsity`: Final sparisty for each layer at `end_epoch`
- `start_epoch`: The epoch to start the OBS algorithm
- `end_epoch`: The epoch to end the OBS algorithm
- `update_frequency`: Frequency at which weights are pruned
- `params`: A list of full parameter names or regest patters to apply pruning to. Regex pattens must be specified with the prefix `re:` (e.g. `"re:.*weight"`). `__ALL__` will match all parameters. `__ALL_PRUNABLE__` will match to all ConvNd and Linear layers' weights.
- `inter_func`: The type of interpolation function to use, which determines how sparsity is added to the network. Options are `linear`, `cubic`, and `inverse_cubic`. `cubic` is the default.
- `mask_type`: String to define the type of sparsity to apply. Options are `unstructured` for unstructured pruning or `4block` for four block pruning or a list of two integers for a custom block shape. In almost all cases, you should use `unstructured`.
- `leave_enabled`: is a boolean value which specifies whether to maintain sparsity after the `end_epoch`. This should generally be set to `True` (the default).
- `global_sparsity`: A boolean indicating whether to apply global pruning (such that the averge sparsity for each layer will equal `final_sparsity`) or layer-wise pruning (such that the sparsity
of each layer is equal to `final_sparsity`). Default is True.
- `num_grads`: number of gradients used to calculate the Fisher approximation, defaults to 1024
- `damp`: dampening factor, default is 1e-7
- `fisher_block_size`: size of blocks along the main diagonal of the Fisher approximation, default is 50
- `grad_sampler_kwargs`: kwargs to override default train dataloader config for pruner's gradient sampling.
- `num_recomputations`: number of recomputations of the inverse Hessian approximation while performing one pruning step

Example:

```yaml
num_pruning_active_epochs: 25
mask_type: "unstructured"
global_sparsity: True

training_modifiers:
   - !OBSPruningModifier
     init_sparsity: 0.05
     final_sparsity: 0.9
     start_epoch: 0.0
     end_epoch: eval(num_pruning_active_epochs)
     update_frequency: 1.0
     params: __ALL_PRUNABLE__
     leave_enabled: True
     inter_func: cubic
     global_sparsity: eval(global_sparsity)
     mask_type: eval(mask_type)
```

[Source Code](https://github.com/neuralmagic/sparseml/blob/main/src/sparseml/pytorch/sparsification/pruning/modifier_pruning_obs.py)

#### MFAC Pruning Modifier

`MFACPruningModifier` implements the [M-FAC algorithm](https://arxiv.org/pdf/2107.03356.pdf). Similiar to OBS, M-FAC is a second-order method for identifying the least important weights. OBS expands upon the initial success of M-FAC, and should be used in most cases.

Parameters:
- `init_sparsity`: Initial sparsity for each layer at `start_epoch`
- `final_sparsity`: Final sparisty for each layer at `end_epoch`
- `start_epoch`: The epoch to start the OBS algorithm
- `end_epoch`: The epoch to end the OBS algorithm
- `update_frequency`: Frequency at which weights are pruned
- `params`: A list of full parameter names or regest patters to apply pruning to. Regex pattens must be specified with the prefix `re:` (e.g. `"re:.*weight"`). `__ALL__` will match all parameters. `__ALL_PRUNABLE__` will match to all ConvNd and Linear layers' weights.
- `inter_func`: The type of interpolation function to use, which determines how sparsity is added to the network. Options are `linear`, `cubic`, and `inverse_cubic`. `cubic` is the default.
- `mask_type`: String to define the type of sparsity to apply. Options are `unstructured` for unstructured pruning or `4block` for four block pruning or a list of two integers for a custom block shape. In almost all cases, you should use `unstructured`.
- `leave_enabled`: is a boolean value which specifies whether to maintain sparsity after the `end_epoch`. This should generally be set to `True` (the default).
- `global_sparsity`: A boolean indicating whether to apply global pruning (such that the averge sparsity for each layer will equal `final_sparsity`) or layer-wise pruning (such that the sparsity of each layer is equal to `final_sparsity`). Default is True.
- `num_grads`: number of gradients used to calculate the Fisher approximation, defaults to 64
- `damp`: dampening factor, default is 1e-5
- `fisher_block_size`: optional value to enable blocked computation of the Fisher matrix. Blocks will be formed consecutively along the diagonal. If None, blocked computation is not used. Default is 2000
- `grads_device`: 
- `num_pages`: number of pages to break the gradient samples into for GPU computation. Only available when blocked computation is not enabled. Default is 1
- `avilable_devices`: list of device names to perform computation on. Default is empty
- `grad_sampler_kwargs`: kwargs to override default train dataloader config for pruner's gradient sampling.

Example:

```yaml
mask_type: unstructured
global_sparsity: True
num_pruning_epochs: 10.0

pruning_modifiers:
  - !MFACPruningModifier
    start_epoch: 0.0
    params: __ALL_PRUNABLE__
    init_sparsity: 0.05
    final_sparsity: 0.8
    end_epoch: eval(num_pruning_epochs)
    update_frequency: 1.0
    mask_type: eval(mask_type)
    global_sparsity: eval(global_sparsity)
    leave_enabled: True
    inter_func: cubic
    num_grads: 64
    fisher_block_size: 10000
```

[Source Code](https://github.com/neuralmagic/sparseml/blob/main/src/sparseml/pytorch/sparsification/pruning/modifier_pruning_mfac.py)

#### MovementPruning

`MovementPruning` implements the [Movement Pruning](https://arxiv.org/abs/2005.07683). Just like gradual magnitude pruning, the Movement Pruning iteratively prunes the least important weights at the end of each epoch. However, rather than removing weights closest to 0 (like GMP), the Movement Pruning is a first-order method that uses a combination of gradient values and magnitude to determine the least important weights. This method can be more effective, as weights moving quickly away from 0 are kept in the network.

Parameters:
- `init_sparsity`: Initial sparsity for each layer at `start_epoch`
- `final_sparsity`: Final sparisty for each layer at `end_epoch`
- `start_epoch`: The epoch to start the GMP algorithm
- `end_epoch`: The epoch to end the GMP algorithm
- `update_frequency`: Frequency at which weights are pruned
- `params`: A list of full parameter names or regest patters to apply pruning to. Regex pattens must be specified with the prefix `re:` (e.g. `"re:.*weight"`). `__ALL__` will match all parameters. `__ALL_PRUNABLE__` will match to all ConvNd and Linear layers' weights.
- `inter_func`: The type of interpolation function to use, which determines how sparsity is added to the network. Options are `linear`, `cubic`, and `inverse_cubic`. `cubic` is the default.
- `mask_type`: String to define the type of sparsity to apply. Options are `unstructured` for unstructured pruning or `4block` for four block pruning or a list of two integers for a custom block shape. In almost all cases, you should use `unstructured`.
- `leave_enabled`: A boolean value which specifies whether to maintain sparsity after the `end_epoch`. This should generally be set to `True` (the default).

Example:

```yaml
training_modifiers:
   - !MovementPruningModifier
     init_sparsity: 0.05
     final_sparsity: 0.8
     start_epoch: 0.0
     end_epoch: 10.0
     update_frequency: 1.0
     params: __ALL_PRUNABLE__
     leave_enabled: True
     inter_func: cubic
     mask_type: unstructured
```

[Source Code](https://github.com/neuralmagic/sparseml/blob/main/src/sparseml/pytorch/sparsification/pruning/modifier_pruning_movement.py)


### Additional Methods

Beyond the iterative pruning methods described above, SparseML also implements other algorithsm for pruning models, such as AC/DC.

#### ACDCPruningModifier

`ACDCPruningModifier` implements the [Alternative Compressed/DeCompressed Training of Deep Neural Networks (AC/DC)](https://arxiv.org/pdf/2106.12379.pdf) algorithm. AC/DC performs co-training of sparse and dense models, where the sparsity mask is applied and removed iteratively during the training process. The AC/DC algorithm was used to create the 95% pruned version of ResNet-50 in [SparseZoo](https://sparsezoo.neuralmagic.com/models/cv%2Fclassification%2Fresnet_v1-50%2Fpytorch%2Fsparseml%2Fimagenet%2Fpruned95_quant-none).

Parameters:
- `compression_sparsity`: The sparsity enforced during the compression phase.
- `start_epoch`: The epoch to start the modifier at
- `end_epoch`: The epoch to end the modifier at
- `update_frequency`: The length (in epochs) of compression/decompression phase
- `params`: A list of full parameter names or regest patters to apply pruning to. Regex pattens must be specified with the prefix `re:` (e.g. `"re:.*weight"`). `__ALL__` will match all parameters. `__ALL_PRUNABLE__` will match to all ConvNd and Linear layers' weights.
- `global_sparsity`: A boolean indicating whether to apply global pruning (such that the averge sparsity for each layer will equal `final_sparsity`) or layer-wise pruning (such that the sparsity of each layer is equal to `final_sparsity`). Default is True.
- `mask_type`: String to define the type of sparsity to apply. Options are `unstructured` for unstructured pruning, `4block` for four block pruning or a list of two integers for a custom block shape. In almost all cases, you should use `unstructured` (the default).
- `momentum_buffer_reset`: A boolean to reset momentum buffer before algorithm enters a consecutive decompression phase. According to the paper: "once all weights are re-introduced, it is beneficial to reset to 0 the gradient momentum term of the optimizer; this is particularly useful for the weights that were previously pruned, which would otherwise have stale versions of gradients." Default is True.
- `leave_enabled`: A boolean value which specifies whether to maintain sparsity after the `end_epoch`. This should generally be set to `True` (the default).

Example:
```yaml
num_pruning_epochs: 20.0

pruning_modifiers:
  - !ACDCPruningModifier
    start_epoch: 0.0
    params: __ALL_PRUNABLE__
    end_epoch: eval(num_pruning_epochs)
    update_frequency: 1.0
    mask_type: unstructured
    global_sparsity: True
    leave_enabled: True
    compression_sparsity: 0.8
```

[Source Code](https://github.com/neuralmagic/sparseml/blob/main/src/sparseml/pytorch/sparsification/pruning/modifier_pruning_acdc.py)


## Quantization

Quantization modifiers handle applying [quantization](https://pytorch.org/docs/stable/quantization.html) to models. With Quantization, we shift weights and activations from a high precision format used for training (typically FP32) to a low precision format (typically INT8). 

### Quantization Aware Training

Quantization Aware Training (QAT) is a popular method that allows quantizing a model and applying fine-tuning to restore accuracy degradation caused by quantization. QAT emulates the precision loss of INT8 quantization during training so weights can be
learned to limit any accuracy loss from quantization. 

#### Quantization Modifiers

`QuantizationModifier` sets the model to run with QAT. Once the `QuantizationModifier` is enabled, it cannot be disabled (no `end_epoch`). Quantization zero points are set to be asymmetric for activations and symmetric for weights.

Parameters
- `start_epoch`: The epoch to start the QAT algorithms at
- `ignore`: Optional list of module class names or submodule names not to quantize
- `disable_quantization_observer_epoch`: Epoch to disable updates to the module quantization observers. At this point, quantized weights and zero points will not be updated. Leave None to not disable observers during QAT. Default is None.
- `freeze_bn_stats_epoch`: Epoch to stop the tracking of batch norm stats. Leave None to not stop tracking batch norm stats during QAT. Default is None.
- `model_fuse_fn_name

Example:

```yaml
quantization_modifiers:
 - !QuantizationModifier
     start_epoch: 0.0
```

[Source Code](https://github.com/neuralmagic/sparseml/blob/main/src/sparseml/pytorch/sparsification/quantization/modifier_quantization.py)

## Distillation Modifiers

The distillation modifiers handle adding model distillation to the pruning and quantization workflows enabled by SparseML. [Knowledge distillation](https://en.wikipedia.org/wiki/Knowledge_distillation) is the process of transfering knowledge from a large model to a smaller model by encouraging the smaller model to mimic the output from the teacher. Distillation can be applied in concert with pruning and quantization, helping to reach higher levels of sparsity without accuracy loss.

Neural Magic has had signficant success applying distillation with Transformers and has early success applying with YOLOv5 in object detection.

#### DistillationModifier

`DistillationModifier` adds a knowledge distillation loss to the training objective based, comparing the output of the teacher against the output of the student. This technique has been heavily used by Neural Magic in pruning transformer models and fine-tuning them onto downstream tasks.

Parameters:
- `start_epoc`: The epoch to start the modifier at
- `end_epoch`: The epoch to end the modifier at
- `distill_output_keys`: list of keys for the module outputs to use for distillation if multiple outputs are present. None or empty list defaults to using all available outputs
- `teacher_input_keys`: list of keys to filter the inputs by before passing into the teacher. None or empty list defaults to using all available inputs
- `hardness`: how much to weight the distillation loss vs the base loss (e.g. hardness of 0.6 will return 0.6 * distill_loss + 0.4 * base_loss). Default is 0.5
- `temperature`: temperature applied to teacher and student softmax for distillation

Example (Transformers):

```yaml
distill_hardness: &distill_hardness 1.0
distill_temperature: &distill_temperature 2.0

distillation_modifiers:
  - !DistillationModifier
     hardness: eval(distill_hardness)
     temperature: eval(distill_temperature)
     distill_output_keys: [logits]
```

[Source Code](https://github.com/neuralmagic/sparseml/blob/main/src/sparseml/pytorch/sparsification/distillation/modifier_distillation.py)

#### PerLayerDistillationModifier

`PerLayerDistillationModifier` adds a knowledge distillation loss to the training objective based on the feature imitation loss. The feature difference between teacher and student can be weighted spatially by a weighing function. This technique has been used to prune the [YOLOv5 models in SparseZoo](https://sparsezoo.neuralmagic.com/models/cv%2Fdetection%2Fyolov5-s%2Fpytorch%2Fultralytics%2Fcoco%2Fpruned75_quant-none).

Parameters:
- `start_epoch`: The epoch to start the distillation at
- `end_epoch`: The epoch to end the distillation at
- `update_frequency`: The number of epochs or fraction of epochs to update at between start and end
- `gain`: How much to weight the distillation loss. Default is `1.5`
- `normalize`: Whether to normalize the output difference by the the magnitude of the teacher's output. Default is `True`.
- `student_layer_names`: List of layer names to distill. *Must be same length as teacher_layer_names.*
- `teacher_layer_names`: List of layer names to distill from. *Must be same length as student_layer_names.* 
- `project_features`: Whether to project the output of student layers to the same size as the output of the teacher layers. Default is `True`
- `epsilon`: Small value used to avoid division by zero when normalization is used. Default is `1e-6`

The connection between layers in the student model and layers in the teacher model is controlled via the `teacher_layer_names` and `student_layer_names` fields. **These fields must be the same length!** The i'th item in both of these arrays are paired together. If either `teacher_layer_names` or `student_layer_names` is `None`, we use the same names for both.

Example (from YOLOv5s):
```
per_layer_distillation_gain: 0.01
num_epochs: 100
knowledge_disitillation_modifiers:
  - !PerLayerDistillationModifier
    start_epoch: 0.0
    end_epoch: eval(num_epochs)
    gain: eval(per_layer_distillation_gain)
    project_features: true
    student_layer_names:
    - model.0
    - model.1
    - model.2.cv1
    - model.2.cv2
    - model.2.cv3
    - model.2.m.0.cv1
    - model.2.m.0.cv2
    - model.3
    - model.4.cv1
    - model.4.cv2
    - model.4.cv3
    - model.4.m.0.cv1
    - model.4.m.0.cv2
    - model.5
    - model.6.cv1
    - model.6.cv2
    - model.6.cv3
    - model.6.m.0.cv1
    - model.6.m.0.cv2
    - model.7
    - model.8.cv1
    - model.8.cv2
    - model.8.cv3
    - model.8.m.0.cv1
    - model.8.m.0.cv2
    - model.9.cv1
    - model.9.cv2
    - model.10
    - model.13.cv1
    - model.13.cv2
    - model.13.cv3
    - model.13.m.0.cv1
    - model.13.m.0.cv2
    - model.14
    - model.17.cv1
    - model.17.cv2
    - model.17.cv3
    - model.17.m.0.cv1
    - model.17.m.0.cv2
    - model.18
    - model.20.cv1
    - model.20.cv2
    - model.20.cv3
    - model.20.m.0.cv1
    - model.20.m.0.cv2
    - model.21
    - model.23.cv1
    - model.23.cv2
    - model.23.cv3
    - model.23.m.0.cv1
    - model.23.m.0.cv2
    - model.24.m.0
    - model.24.m.1
    - model.24.m.2
```

[Source Code](https://github.com/neuralmagic/sparseml/blob/main/src/sparseml/pytorch/sparsification/distillation/modifier_per_layer.py)
