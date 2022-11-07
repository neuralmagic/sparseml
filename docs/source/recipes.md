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

# Sparsification Recipes

All SparseML Sparsification APIs are designed to work with recipes.
The files encode the instructions needed for modifying the model and/or training process as a list of modifiers.
Example modifiers can be anything from setting the learning rate for the optimizer to gradual magnitude pruning.
The files are written in [YAML](https://yaml.org/) and stored in YAML or 
[markdown](https://www.markdownguide.org/) files using 
[YAML front matter.](https://assemble.io/docs/YAML-front-matter.html)
The rest of the SparseML system is coded to parse the recipe files into a native format for the desired framework
and apply the modifications to the model and training pipeline.

In a recipe, modifiers must be written in a list that includes "modifiers" in its name.

The easiest ways to get or create recipes are by either using the pre-configured recipes in [SparseZoo](https://github.com/neuralmagic/sparsezoo) or using [Sparsify's](https://github.com/neuralmagic/sparsify) automatic creation.

However, power users may be inclined to create their own recipes by hand to enable more 
fine-grained control or to add in custom modifiers.

A sample recipe for pruning a model generally looks like the following:
```yaml
version: 0.1.0
modifiers:
    - !EpochRangeModifier
        start_epoch: 0.0
        end_epoch: 70.0

    - !LearningRateModifier
        start_epoch: 0
        end_epoch: -1.0
        update_frequency: -1.0
        init_lr: 0.005
        lr_class: MultiStepLR
        lr_kwargs: {'milestones': [43, 60], 'gamma': 0.1}

    - !GMPruningModifier
        start_epoch: 0
        end_epoch: 40
        update_frequency: 1.0
        init_sparsity: 0.05
        final_sparsity: 0.85
        mask_type: unstructured
        params: ['sections.0.0.conv1.weight', 'sections.0.0.conv2.weight', 'sections.0.0.conv3.weight']
```

## Modifiers Intro

Recipes can contain multiple modifiers, each modifying a portion of the training process in a different way.
In general, each modifier will have a start and an end epoch for when the modifier should be active.
The modifiers will start at `start_epoch` and run until `end_epoch`.
Note that it does not run through `end_epoch`.
Additionally, all epoch values support decimal values such that they can be started somewhere in the middle of an epoch. 
For example, `start_epoch: 2.5` will start in the middle of the second training epoch.

The most commonly used modifiers are enumerated as subsections below.

## Training Epoch Modifiers

The `EpochRangeModifier` controls the range of epochs for training a model.
Each supported ML framework has an implementation to enable easily retrieving this number of epochs.
Note, that this is not a hard rule and if other modifiers have a larger `end_epoch` or smaller `start_epoch`
then those values will be used instead.

The only parameters that can be controlled for `EpochRangeModifier` are the `start_epoch` and `end_epoch`.
Both parameters are required.

Required Parameters:

- `start_epoch`: The start range for the epoch (0 indexed)
- `end_epoch`: The end range for the epoch

 Example:

 ```yaml
     - !EpochRangeModifier
        start_epoch: 0.0
        end_epoch: 25.0
 ```

## Pruning Modifiers

The pruning modifiers handle [pruning](https://neuralmagic.com/blog/pruning-overview/) 
the specified layer(s) in a given model.

### ConstantPruningModifier

The `ConstantPruningModifier` enforces the sparsity structure and level for an already pruned layer(s) in a model.
The modifier is generally used for transfer learning from an already pruned model or 
to enforce sparsity while quantizing.
The weights remain trainable in this setup; however, the sparsity is unchanged.

Required Parameters:
- `params`: The parameters in the model to prune.
   This can be set to a string containing `__ALL__` to prune all parameters, a list to specify the targeted parameters,
   or regex patterns prefixed by 're:' of parameter name patterns to match.
   For example: `['blocks.1.conv']` for PyTorch and `['mnist_net/blocks/conv0/conv']` for TensorFlow.
   Regex can also be used to match all conv params: `['re:.*conv']` for PyTorch and `['re:.*/conv']` for TensorFlow.

Example:

```yaml
    - !ConstantPruningModifier
        params: __ALL__
```

#### GMPruningModifier

The `GMPruningModifier` prunes the parameter(s) in a model to a 
target sparsity (percentage of 0's for a layer's param/variable) 
using [gradual magnitude pruning.](https://neuralmagic.com/blog/pruning-gmp/)
This is done gradually from an initial to final sparsity (`init_sparsity`, `final_sparsity`)
over a range of epochs (`start_epoch`, `end_epoch`) and updated at a specific interval defined by the `update_frequency`.
For example, using the following settings `start_epoch: 0`, `end_epoch: 5`, `update_frequency: 1`, 
`init_sparsity: 0.05`, `final_sparsity: 0.8` will do the following:
- at epoch 0 set the sparsity for the specified param(s) to 5%
- once every epoch, gradually increase the sparsity towards 80%
- by the start of epoch 5, stop pruning and set the final sparsity for the specified param(s) to 80%

Required Parameters:

- `params`: The parameters in the model to prune. 
   This can be set to a string containing `__ALL__` to prune all parameters, a list to specify the targeted parameters,
   or regex patterns prefixed by 're:' of parameter name patterns to match.
   For example: `['blocks.1.conv']` for PyTorch and `['mnist_net/blocks/conv0/conv']` for TensorFlow.
   Regex can also be used to match all conv params: `['re:.*conv']` for PyTorch and `['re:.*/conv']` for TensorFlow.
- `init_sparsity`: The decimal value for the initial sparsity to start pruning with.
   At `start_epoch` will set the sparsity for the param/variable to this value. 
   Generally, this is kept at 0.05 (5%).
- `final_sparsity`: The decimal value for the final sparsity to end pruning with.
   By the start of `end_epoch` will set the sparsity for the param/variable to this value.
   Generally, this is kept in a range from 0.6 to 0.95 depending on the model and layer. 
   Anything less than 0.4 is not useful for performance.
- `start_epoch`: The epoch to start the pruning at (0 indexed).
   This supports floating-point values to enable starting pruning between epochs.
 - `end_epoch`: The epoch before which to stop pruning.
   This supports floating-point values to enable stopping pruning between epochs.
 - `update_frequency`: The number of epochs/fractions of an epoch between each pruning step.
   It supports floating-point values to enable updating inside of epochs.
   Generally, this is set to update once per epoch (`1.0`). 
   However, if the loss for the model recovers quickly, it should be set to a lesser value.
   For example: set it to `0.5` for once every half epoch (twice per epoch).

Example:

```yaml
    - !GMPruningModifier
        params: ['blocks.1.conv']
        init_sparsity: 0.05
        final_sparsity: 0.8
        start_epoch: 5.0
        end_epoch: 20.0
        update_frequency: 1.0
```

### Quantization Modifiers

The `QuantizationModifier` sets the model to run with 
[quantization aware training (QAT).](https://pytorch.org/docs/stable/quantization.html) 
QAT emulates the precision loss of int8 quantization during training so weights can be
learned to limit any accuracy loss from quantization. 
Once the `QuantizationModifier` is enabled, it cannot be disabled (no `end_epoch`). 
Quantization zero points are set to be asymmetric for activations and symmetric for weights. 
Currently only available in PyTorch.

Notes:
- ONNX exports of PyTorch QAT models will be QAT models themselves (emulated quantization).
   To convert your QAT ONNX model to a fully quantizerd model use
   the script `scripts/pytorch/model_quantize_qat_export.py` or the function
   `neuralmagicML.pytorch.quantization.quantize_qat_export`.
- If performing QAT on a sparse model, you must preserve sparsity during QAT by
   applying a `ConstantPruningModifier` or have already used a `GMPruningModifier` with
   `leave_enabled` set to True.

Required Parameters:

- `start_epoch`: The epoch to start QAT. This supports floating-point values to enable
  starting pruning between epochs.

Example:

```yaml
    - !QuantizationModifier
        start_epoch: 0.0
```

### Learning Rate Modifiers

The learning rate modifiers set the learning rate (LR) for an optimizer during training.
If you are using an Adam optimizer, then generally, these are not useful.
If you are using a standard stochastic gradient descent optimizer, then these give a convenient way to control the LR.

#### SetLearningRateModifier

The `SetLearningRateModifier` sets the learning rate (LR) for the optimizer to a specific value at a specific point
in the training process.

Required Parameters:

- `start_epoch`: The epoch in the training process to set the `learning_rate` value for the optimizer.
   This supports floating-point values to enable setting the LR between epochs.
- `learning_rate`: The floating-point value to set as the learning rate for the optimizer at `start_epoch`.

Example:

```yaml
    - !SetLearningRateModifier
        start_epoch: 5.0
        learning_rate: 0.1
```

#### LearningRateModifier

The `LearningRateModifier` sets schedules for controlling the learning rate for an optimizer during training.
If you are using an Adam optimizer, then generally, these are not useful.
If you are using a standard stochastic gradient descent optimizer, then these give a convenient way to control the LR.
Provided schedules to choose from are the following:

- `ExponentialLR`: Multiplies the learning rate by a `gamma` value every epoch.
   To use this one, `lr_kwargs` should be set to a dictionary containing `gamma`.
   For example: `{'gamma': 0.9}`
- `StepLR`: Multiplies the learning rate by a `gamma` value after a certain epoch period defined by `step`.
   To use this one, `lr_kwargs` must be set to a dictionary containing `gamma` and `step_size`.
   For example: `{'gamma': 0.9, 'step_size': 2.0}`
- `MultiStepLR`: Multiplies the learning rate by a `gamma` value at specific epoch points defined by `milestones`.
   To use this one, `lr_kwargs` must be set to a dictionary containing `gamma` and `milestones`.
   For example: `{'gamma': 0.9, 'milestones': [2.0, 5.5, 10.0]}`

Required Parameters:

- `start_epoch`: The epoch to start modifying the LR at (0 indexed).
   This supports floating-point values to enable starting pruning between epochs.
- `end_epoch`: The epoch to stop modifying the LR before.
   This supports floating-point values to enable stopping pruning between epochs. 
- `lr_class`: The LR class to use, one of [`ExponentialLR`, `StepLR`, `MultiStepLR`].
- `lr_kwargs`: The named arguments for the `lr_class`.
- `init_lr`: [Optional] The initial LR to set at `start_epoch` and to use for creating the schedules. 
   If not given, the optimizer's current LR will be used at startup.

 Example:

 ```yaml
     - !LearningRateModifier
        start_epoch: 0.0
        end_epoch: 25.0
        lr_class: MultiStepLR
        lr_kwargs:
            gamma: 0.9
            milestones: [2.0, 5.5, 10.0]
        init_lr: 0.1
 ```

### Params/Variables Modifiers

#### TrainableParamsModifier

The `TrainableParamsModifier` controls the params that are marked as trainable for the current optimizer.
This is generally useful when transfer learning to easily mark which parameters should or should not be frozen/trained.

Required Parameters:

- `params`: The names of parameters to mark as trainable or not.
    This can be set to a string containing `__ALL__` to mark all parameters, a list to specify the targeted parameters,
    or regex patterns prefixed by 're:' of parameter name patterns to match.
    For example: `['blocks.1.conv']` for PyTorch and `['mnist_net/blocks/conv0/conv']` for TensorFlow.
    Regex can also be used to match all conv params: `['re:.*conv']` for PyTorch and `['re:.*/conv']` for TensorFlow.

Example:

```yaml
    - !TrainableParamsModifier
      params: __ALL__
```

### Optimizer Modifiers

#### SetWeightDecayModifier

The `SetWeightDecayModifier` sets the weight decay (L2 penalty) for the optimizer to a
specific value at a specific point in the training process.

Required Parameters:

- `start_epoch`: The epoch in the training process to set the `weight_decay` value for the
    optimizer. This supports floating-point values to enable setting the weight decay
    between epochs.
- `weight_decay`: The floating-point value to set as the weight decay for the optimizer
    at `start_epoch`.

Example:

```yaml
    - !SetWeightDecayModifier
        start_epoch: 5.0
        weight_decay: 0.0
```
