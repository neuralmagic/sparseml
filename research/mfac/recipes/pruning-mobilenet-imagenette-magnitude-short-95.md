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
# Epoch Variables
start_epoch: &start_epoch 0.0
end_epoch: &end_epoch 3.0
pruning_start_epoch: &pruning_start_epoch 1.0
pruning_end_epoch: &pruning_end_epoch 2.0
lr: &lr 0.0004

# Pruning Variables
pruning_update_frequency: &pruning_update_frequency 1.0
pruning_mask_type: &pruning_mask_type unstructured

target_sparsities: &target_sparsities
  0.90: ['sections.1.0.point.conv.weight', 'sections.1.1.point.conv.weight', 'sections.2.0.point.conv.weight']
  0.95: ['sections.2.1.point.conv.weight', 'sections.3.0.point.conv.weight', 'sections.3.1.point.conv.weight', 'sections.3.5.point.conv.weight']
  0.97: ['sections.3.2.point.conv.weight', 'sections.3.3.point.conv.weight', 'sections.3.4.point.conv.weight', 'sections.4.0.point.conv.weight', 'sections.4.1.point.conv.weight']

# Modifiers Groups:
training_modifiers:
  - !EpochRangeModifier
    start_epoch: *start_epoch
    end_epoch: *end_epoch

  - !SetLearningRateModifier
    start_epoch: *start_epoch
    learning_rate: *lr

pruning_modifiers:
  - !GMPruningModifier
    params: []
    init_sparsity: 0.35
    final_sparsity: *target_sparsities
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    update_frequency: *pruning_update_frequency
    mask_type: *pruning_mask_type

  - !SetWeightDecayModifier
    weight_decay: 0.0
    start_epoch: *pruning_end_epoch
---

# Pruning MobileNet-Imagenette Magnitude
This recipe prunes a MobileNet model to 95% sparsity over 3 epochs using magnitude pruning.