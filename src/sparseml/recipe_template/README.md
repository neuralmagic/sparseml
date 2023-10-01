# Recipe Template

`recipe-template` can be used to generate recipes for pruning, quantization, and distillation. The generated recipe can then be used as-is with sparseml or easily altered to fit the user's needs. `recipe-template` can be used as a CLI tool or as a Python API.

## Requirements

`sparseml[torchvision]>=1.3`

### CLI

`sparseml.recipe_template` is a CLI tool that can be used to generate a recipe based on specified options. The generated recipe is outputted to a markdown file. Invoke the CLI with the `--help` flag to see the available options.

`sparseml.recipe_template --help`

```bash
$ sparseml.recipe_template --help
Usage: sparseml.recipe_template [OPTIONS]

 Utility to create a recipe template based on specified options

 Example for using sparseml.pytorch.recipe_template:

 `sparseml.recipe_template --pruning true --quantization true`

 `sparseml.recipe_template --quantization true --target vnni --lr
 constant `

Options:
 --version Show the version and exit.
 --pruning [true|false|gmp|acdc|mfac|movement|constant]
 Specify if recipe should include pruning
 steps, can also take in the name of a
 pruning algorithm [default: false]
 --quantization [true|false] Specify if recipe should include
 quantization steps, can also take in the
 name of the target hardware [default:
 false]
 --lr, --learning_rate [constant|cyclic|stepped|exponential|linear]
 Specify learning rate schedule function
 [default: constant]
 --target [vnni|tensorrt|default]
 Specify target hardware type for current
 recipe [default: default]
 --distillation Add distillation support to the recipe
 --file_name TEXT The file name to output recipe to [default:recipe.md]
 --help Show this message and exit.
```

Example CLI Usage:

1. Generate a recipe for pruning

 ```bash
 sparseml.recipe_template --pruning true
 ```

2. Generate a recipe for quantization

 ```bash
 sparseml.recipe_template --quantization true
 ```

3. Generate a recipe for pruning and quantization

 ```bash
 sparseml.recipe_template --pruning true --quantization true
 ```

4. Generate a recipe for pruning and quantization with a learning rate schedule

 ```bash
 sparseml.recipe_template --pruning true --quantization true --lr cyclic
 ```

5. Generate a recipe for pruning and quantization with a cyclic learning rate schedule and distillation

 ```bash
 sparseml.recipe_template --pruning true --quantization true --lr cyclic --distillation
 ```

 Generated recipe:

 ```text
 Template:
 ---
 lr_func: cyclic
 init_lr: 0.001
 final_lr: 0.0
 num_epochs: 20.0
 num_qat_epochs: 5.0
 num_qat_finetuning_epochs: 2.5
 quantization_submodules: null
 num_pruning_active_epochs: 7.5
 num_pruning_finetuning_epochs: 7.5
 pruning_init_sparsity: 0.05
 pruning_final_sparsity: 0.8
 pruning_update_frequency: 0.375
 mask_type: unstructured
 global_sparsity: True
 _num_pruning_epochs: eval(num_pruning_active_epochs + num_pruning_finetuning_epochs)
 distillation_hardness: 0.5
 distillation_temperature: 2.0

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

 quantization_modifiers:
 - !QuantizationModifier
 start_epoch: eval(_num_pruning_epochs)
 submodules: eval(quantization_submodules)
 disable_quantization_observer_epoch: eval(_num_pruning_epochs + num_qat_epochs - num_qat_finetuning_epochs)
 freeze_bn_stats_epoch: eval(_num_pruning_epochs + num_qat_epochs - num_qat_finetuning_epochs)
 tensorrt: False
 quantize_linear_activations: False
 quantize_conv_activations: False
 quantize_embedding_activations: False
 exclude_module_types: 
 - LayerNorm
 - Tanh

 distillation_modifiers:
 - !DistillationModifier
 start_epoch: 0.0
 end_epoch: eval(num_epochs)
 hardness: eval(distillation_hardness)
 temperature: eval(distillation_temperature)
 ---
 This recipe defines the hyperparams necessary to apply specified sparsification 
 techniques.
 
 Users are encouraged to experiment with the training length and initial learning 
 rate to either expedite training or to produce a more accurate model.

 This can be done by either editing the recipe or supplying the `--recipe_args` 
 argument to the training commands.

 For example, the following appended to the training commands will change the number 
 of epochs and the initial learning rate:

 ```bash
 --recipe_args '{"num_epochs":8,"init_lr":0.0001}'
 ```

 The following template can be used to apply this recipe in a `one-shot` manner:

 ```python
 from sparseml.pytorch.optim import ScheduledModifierManager

 model = ... # instantiated model
 recipe = ... # local path to this recipe

 manager = ScheduledModifierManager.from_yaml(recipe)
 manager.apply(model) 
 ```

## Python API

`recipe_template` also provides a Python API for generating recipes. The API is more powerful than the CLI and allows for more customization. The API can be used to generate recipes for pruning, quantization, and distillation.

Additionally, the API can also consume instantiated PyTorch models and automatically generate a recipe based on the model's architecture. This is useful for generating tailored recipes specific to the model. The generated recipe can then be used to sparsify the models as-is or easily altered to fit the user's needs.

```python
>>> # Instantiated Model
>>> from sparseml.pytorch.models import resnet18
>>> model = resnet18()
>>> 
>>> # import and use recipe_template
>>> from sparseml.pytorch import recipe_template
>>> 
>>> # Generate a pruning and quantization recipe 
>>> # for instantiated model
>>> print(recipe_template(model=model, pruning="acdc", quantization=True, distillation=True))
lr_func: linear
init_lr: 0.001
final_lr: 0.0
num_epochs: 20.0
num_qat_epochs: 5.0
num_qat_finetuning_epochs: 2.5
quantization_submodules: null
num_pruning_active_epochs: 7.5
num_pruning_finetuning_epochs: 7.5
pruning_init_sparsity: 0.05
pruning_final_sparsity: 0.8
pruning_update_frequency: 0.375
mask_type: unstructured
global_sparsity: True
_num_pruning_epochs: eval(num_pruning_active_epochs + num_pruning_finetuning_epochs)
distillation_hardness: 0.5
distillation_temperature: 2.0

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
 - !ACDCPruningModifier
 start_epoch: 0.0
 params: 
 - sections.0.0.conv1.weight
 - sections.0.0.conv2.weight
 - sections.0.1.conv1.weight
 - sections.0.1.conv2.weight
 - sections.1.0.conv1.weight
 - sections.1.0.conv2.weight
 - sections.1.0.identity.conv.weight
 - sections.1.1.conv1.weight
 - sections.1.1.conv2.weight
 - sections.2.0.conv1.weight
 - sections.2.0.conv2.weight
 - sections.2.0.identity.conv.weight
 - sections.2.1.conv1.weight
 - sections.2.1.conv2.weight
 - sections.3.0.conv1.weight
 - sections.3.0.conv2.weight
 - sections.3.0.identity.conv.weight
 - sections.3.1.conv1.weight
 - sections.3.1.conv2.weight
 - classifier.fc.weight
 end_epoch: eval(num_pruning_active_epochs)
 update_frequency: 1.0
 mask_type: eval(mask_type)
 global_sparsity: eval(global_sparsity)
 leave_enabled: True
 compression_sparsity: 0.8

quantization_modifiers:
 - !QuantizationModifier
 start_epoch: eval(_num_pruning_epochs)
 submodules: eval(quantization_submodules)
 disable_quantization_observer_epoch: eval(_num_pruning_epochs + num_qat_epochs - num_qat_finetuning_epochs)
 freeze_bn_stats_epoch: eval(_num_pruning_epochs + num_qat_epochs - num_qat_finetuning_epochs)
 tensorrt: False
 quantize_linear_activations: False
 quantize_conv_activations: False
 quantize_embedding_activations: False
 exclude_module_types: 
 - LayerNorm
 - Tanh

distillation_modifiers:
 - !DistillationModifier
 start_epoch: 0.0
 end_epoch: eval(num_epochs)
 hardness: eval(distillation_hardness)
 temperature: eval(distillation_temperature)
```

Refer to the docstrings for more information on the API
here: [recipe_template](https://github.com/neuralmagic/sparseml/blob/c610973faedec318e5b7df81f04bbf1e2e6ac532/src/sparseml/pytorch/recipe_template/main.py#L67-L68)



<!-- Anchors -->
[SparseML]: https://neuralmagic.com/sparseml/
