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
start_epoch: &start_epoch 0
start_epoch_ac_dc: &start_epoch_ac_dc 5
end_epoch: &end_epoch 100
learning_rate: &learning_rate 0.256
compression_sparsity: &compression_sparsity 0.9
global_sparsity: &global_sparsity True
params: &params ['re:.*conv*', 're:.*fc.weight*']

training_modifiers:
  - !EpochRangeModifier
    start_epoch: *start_epoch
    end_epoch: *end_epoch
  
  - !LearningRateFunctionModifier
    start_epoch: *start_epoch
    end_epoch: *start_epoch_ac_dc
    lr_func: linear
    init_lr: 0.0512
    final_lr: *learning_rate

  - !LearningRateFunctionModifier
    start_epoch: *start_epoch_ac_dc
    end_epoch: *end_epoch
    lr_func: cosine
    init_lr: *learning_rate
    final_lr: 0.0
  
pruning_modifiers:
  - !LegacyGMPruningModifier 
       init_sparsity: *compression_sparsity
       final_sparsity: *compression_sparsity
       start_epoch: 85
       end_epoch: *end_epoch
       update_frequency: 1
       params: *params
       global_sparsity: *global_sparsity

  - !ACDCPruningModifier
      compression_sparsity: *compression_sparsity
      start_epoch: *start_epoch_ac_dc
      end_epoch: 75
      update_frequency: 5
      params: *params
      global_sparsity: *global_sparsity
---
```
Can we use something better than LegacyGMPruningModifier
for final sparse fine-tuning?
Something like ConstantPruningModifier, but the one which
enforces its own applied_sparsity (instead of holding previous
sparsity level).
```

