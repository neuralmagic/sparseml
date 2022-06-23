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

---
version: 1.1.0

# General Params
num_epochs: 3
init_lr: 1.5e-4 
final_lr: 0
lr_rewind_epoch: 2

# Distillation Params
distill_hardness: 1.0
distill_temperature: 2.0

# Pruning Params
pruning_init_sparsity: 0.7
pruning_final_sparsity: 0.9
pruning_start_epoch: 0
pruning_end_epoch: 1
pruning_update_frequency: 0.01

# Quantization Params
qat_epochs: 1
qat_no_observer_epochs: 1.0
quantize_embeddings: 1


# Modifiers:
training_modifiers:
  - !EpochRangeModifier
      end_epoch: eval(num_epochs)
      start_epoch: 0.0
    
  - !LearningRateFunctionModifier
    start_epoch: 0
    end_epoch: eval(lr_rewind_epoch)
    lr_func: linear
    init_lr: eval(init_lr)
    final_lr: eval(final_lr)

distillation_modifiers:
  - !DistillationModifier
     hardness: eval(distill_hardness)
     temperature: eval(distill_temperature)
     distill_output_keys: [start_logits, end_logits]

pruning_modifiers:
  - !MagnitudePruningModifier
    params: __ALL_PRUNABLE__
    start_epoch: eval(pruning_start_epoch)
    end_epoch: eval(pruning_end_epoch)
    init_sparsity: eval(pruning_init_sparsity)
    final_sparsity: eval(pruning_final_sparsity)
    inter_func: cubic
    update_frequency: eval(pruning_update_frequency)
    mask_type: [1, 4]

quantization_modifiers:
  - !QuantizationModifier
      start_epoch: eval(num_epochs - qat_epochs)
      disable_quantization_observer_epoch: eval(num_epochs - qat_no_observer_epochs)
      freeze_bn_stats_epoch: eval(num_epochs - qat_no_observer_epochs)
      quantize_embeddings: eval(quantize_embeddings)
      quantize_linear_activations: 0
      exclude_module_types: ['LayerNorm', 'Tanh']
      submodules:
        - bert.embeddings
        - bert.encoder
        - qa_outputs
---