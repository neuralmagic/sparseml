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
pruning_modifiers:
  - !GMPruningModifier
    params: __ALL_PRUNABLE__
    init_sparsity: 0.0
    final_sparsity: 0.35
    start_epoch: 0.0
    end_epoch: 1.0
    update_frequency: 1.0
---

# Pruning MNISTNet with Magnitude
This recipe prunes a model to 35% sparsity using magnitude pruning.
This recipe is intended for use with one-shot pruning and is for example
purposes only.